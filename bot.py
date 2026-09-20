import asyncio
import logging
import math
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.methods import SendDocument, SendVideo
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN", "").strip()
BOT_API_BASE = os.getenv("BOT_API_BASE", "").strip().rstrip("/")
BOT_API_LOCAL = os.getenv("BOT_API_LOCAL", "0").strip() == "1"

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg").strip()
WORKDIR = Path(os.getenv("WORKDIR", "./work")).resolve()
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "10000"))
MAX_UNPACKED_GB = Decimal(os.getenv("MAX_UNPACKED_GB", "100"))
MAX_CONCURRENT_RENDERS = int(os.getenv("MAX_CONCURRENT_RENDERS", "1"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "60"))
CRF = int(os.getenv("CRF", "18"))
PRESET = os.getenv("PRESET", "medium").strip()

OFFICIAL_DOWNLOAD_LIMIT = 20 * 1024 * 1024
OFFICIAL_UPLOAD_LIMIT = 50 * 1024 * 1024
LOCAL_UPLOAD_SAFE_LIMIT = 1_950_000_000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

router = Router()
dp = Dispatcher()
dp.include_router(router)

render_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)
sessions: dict[int, "UserSession"] = {}
user_locks: dict[int, asyncio.Lock] = {}


@dataclass
class UserSession:
    root: Path
    frames_dir: Path
    frames: list[Path]
    width: int
    height: int
    source_name: str
    waiting_custom_speed: bool = False
    render_task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)


def get_lock(user_id: int) -> asyncio.Lock:
    lock = user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        user_locks[user_id] = lock
    return lock


def speed_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="0.02 c", callback_data="speed:0.02"),
            InlineKeyboardButton(text="0.04 c", callback_data="speed:0.04"),
            InlineKeyboardButton(text="0.05 c", callback_data="speed:0.05"),
        ],
        [
            InlineKeyboardButton(text="0.10 c", callback_data="speed:0.10"),
            InlineKeyboardButton(text="0.20 c", callback_data="speed:0.20"),
            InlineKeyboardButton(text="0.50 c", callback_data="speed:0.50"),
        ],
        [
            InlineKeyboardButton(text="1.00 c", callback_data="speed:1.00"),
            InlineKeyboardButton(text="Свое значение", callback_data="speed:custom"),
        ],
        [
            InlineKeyboardButton(text="Удалить кадры", callback_data="session:delete"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def done_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Сделать еще с другой скоростью",
                    callback_data="render:again",
                )
            ],
            [
                InlineKeyboardButton(text="Удалить кадры", callback_data="session:delete"),
            ],
        ]
    )


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def fmt_bytes(value: int) -> str:
    size = float(value)
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "Б":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} ТБ"


def fmt_duration(seconds: Decimal) -> str:
    total_ms = int((seconds * 1000).to_integral_value())
    if total_ms < 1000:
        return f"{total_ms} мс"

    total_sec = total_ms / 1000
    if total_sec < 60:
        return f"{total_sec:.2f} сек".rstrip("0").rstrip(".")

    minutes = int(total_sec // 60)
    sec = total_sec - minutes * 60
    if minutes < 60:
        return f"{minutes}:{sec:05.2f}"

    hours = minutes // 60
    minutes %= 60
    return f"{hours}:{minutes:02d}:{sec:05.2f}"


def parse_speed(text: str) -> Decimal:
    value = text.strip().lower().replace(",", ".")
    value = value.replace("сек", "s").replace("с", "s").replace(" ", "")

    if value.endswith("fps"):
        number = Decimal(value[:-3])
        if number <= 0:
            raise ValueError
        result = Decimal(1) / number
    elif value.endswith("мs"):
        result = Decimal(value[:-2]) / Decimal(1000)
    elif value.endswith("ms"):
        result = Decimal(value[:-2]) / Decimal(1000)
    elif value.endswith("s"):
        result = Decimal(value[:-1])
    elif "/" in value:
        left, right = value.split("/", 1)
        denominator = Decimal(right)
        if denominator == 0:
            raise ValueError
        result = Decimal(left) / denominator
    else:
        result = Decimal(value)

    # 0.004 sec = 250 fps. Above that is already mostly codec/player masochism.
    if result < Decimal("0.004") or result > Decimal("60"):
        raise ValueError
    return result


def fps_fraction(frame_duration: Decimal) -> str:
    duration_fraction = Fraction(frame_duration)
    fps = Fraction(1, 1) / duration_fraction
    fps = fps.limit_denominator(1_000_000)
    return f"{fps.numerator}/{fps.denominator}"


def safe_remove(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def extract_and_validate(zip_path: Path, root: Path) -> tuple[list[Path], int, int]:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    max_unpacked = int(MAX_UNPACKED_GB * Decimal(1024**3))

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError("Файл не похож на нормальный ZIP") from exc

    with archive:
        candidates = []
        total_unpacked = 0

        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise ValueError("ZIP с паролем не поддерживается")

            suffix = Path(info.filename).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue

            candidates.append(info)
            total_unpacked += info.file_size

        if not candidates:
            raise ValueError("В ZIP не найдено PNG/JPG/JPEG/WebP/BMP")

        if len(candidates) > MAX_FRAMES:
            raise ValueError(
                f"Кадров {len(candidates):,}, лимит сейчас {MAX_FRAMES:,}".replace(",", " ")
            )

        if total_unpacked > max_unpacked:
            raise ValueError(
                f"После распаковки будет около {fmt_bytes(total_unpacked)}, "
                f"а лимит выставлен {MAX_UNPACKED_GB} ГБ"
            )

        disk = shutil.disk_usage(root)
        reserve = 512 * 1024 * 1024
        if disk.free < total_unpacked + reserve:
            raise ValueError(
                f"Не хватает места на диске. Нужно хотя бы "
                f"{fmt_bytes(total_unpacked + reserve)}, свободно {fmt_bytes(disk.free)}"
            )

        candidates.sort(
            key=lambda i: (
                natural_key(Path(i.filename).name),
                natural_key(i.filename),
            )
        )

        extracted: list[Path] = []

        for index, info in enumerate(candidates, start=1):
            suffix = Path(info.filename).suffix.lower()
            target = frames_dir / f"frame_{index:06d}{suffix}"

            with archive.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

            extracted.append(target)

    width = height = None

    for index, frame in enumerate(extracted, start=1):
        try:
            with Image.open(frame) as im:
                size = im.size
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Кадр {index} поврежден или не является картинкой") from exc

        if width is None:
            width, height = size
        elif size != (width, height):
            raise ValueError(
                f"Размер кадра {index}: {size[0]}x{size[1]}, "
                f"а у первого кадра {width}x{height}. "
                f"Все кадры должны быть одного размера"
            )

    assert width is not None and height is not None
    return extracted, width, height


def make_concat_file(session: UserSession, frame_duration: Decimal) -> Path:
    concat_path = session.root / "frames.ffconcat"

    duration_text = format(frame_duration, "f")
    lines = ["ffconcat version 1.0"]

    for frame in session.frames:
        # Имена кадров создаются нами, поэтому кавычки и прочий цирк в путях исключены.
        lines.append(f"file '{frame.as_posix()}'")
        lines.append(f"duration {duration_text}")

    # concat demuxer не применяет duration к последнему кадру без следующего packet.
    # Дублируем последний, а -t ниже жестко обрезает видео до правильной длины.
    lines.append(f"file '{session.frames[-1].as_posix()}'")

    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return concat_path


async def safe_edit(message: Message, text: str, reply_markup=None) -> None:
    try:
        await message.edit_text(text, reply_markup=reply_markup)
    except TelegramBadRequest:
        pass
    except Exception:
        pass


def build_bot() -> Bot:
    if not TOKEN:
        raise RuntimeError("В .env не указан BOT_TOKEN")

    if BOT_API_BASE:
        api = TelegramAPIServer.from_base(
            BOT_API_BASE,
            is_local=BOT_API_LOCAL,
        )
        session = AiohttpSession(api=api)
        return Bot(TOKEN, session=session)

    return Bot(TOKEN)


async def stop_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def render_and_send(
    bot: Bot,
    chat_id: int,
    user_id: int,
    frame_duration: Decimal,
) -> None:
    session = sessions.get(user_id)
    if session is None:
        await bot.send_message(chat_id, "Кадры уже удалены. Пришли ZIP заново")
        return

    session.last_used = time.monotonic()
    total_duration = frame_duration * len(session.frames)
    fps_text = fps_fraction(frame_duration)

    status = await bot.send_message(
        chat_id,
        (
            f"Рендер поставлен в очередь\n\n"
            f"Кадров: {len(session.frames):,}\n"
            f"Размер: {session.width}x{session.height}\n"
            f"1 кадр: {frame_duration} сек\n"
            f"FPS: {fps_text}\n"
            f"Длина: {fmt_duration(total_duration)}"
        ).replace(",", " "),
    )

    output_path = session.root / f"video_{str(frame_duration).replace('.', '_')}s.mp4"
    concat_path = None
    proc = None
    stderr_task = None

    try:
        async with render_semaphore:
            current = sessions.get(user_id)
            if current is not session:
                await safe_edit(status, "Этот архив уже заменен новым")
                return

            concat_path = make_concat_file(session, frame_duration)
            pix_fmt = (
                "yuv420p"
                if session.width % 2 == 0 and session.height % 2 == 0
                else "yuv444p"
            )

            command = [
                FFMPEG_BIN,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-t",
                format(total_duration, "f"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                PRESET,
                "-crf",
                str(CRF),
                "-r",
                fps_text,
                "-fps_mode",
                "cfr",
                "-pix_fmt",
                pix_fmt,
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(output_path),
            ]

            await safe_edit(
                status,
                (
                    f"Рендер 0%\n\n"
                    f"{len(session.frames):,} кадров - "
                    f"{session.width}x{session.height} - "
                    f"{frame_duration} сек/кадр"
                ).replace(",", " "),
            )

            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(proc.stderr.read())

            last_percent = -1
            last_edit = 0.0
            total_float = float(total_duration)

            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break

                decoded = line.decode("utf-8", "replace").strip()
                if decoded.startswith("out_time_us="):
                    try:
                        out_us = int(decoded.split("=", 1)[1])
                    except ValueError:
                        continue

                    progress = min(99, max(0, int((out_us / 1_000_000) / total_float * 100)))
                    now = time.monotonic()

                    if progress != last_percent and (
                        progress >= last_percent + 5 or now - last_edit >= 5
                    ):
                        await safe_edit(
                            status,
                            (
                                f"Рендер {progress}%\n\n"
                                f"{len(session.frames):,} кадров - "
                                f"{session.width}x{session.height} - "
                                f"{frame_duration} сек/кадр"
                            ).replace(",", " "),
                        )
                        last_percent = progress
                        last_edit = now

            return_code = await proc.wait()
            stderr = b""
            if stderr_task is not None:
                stderr = await stderr_task

            if return_code != 0:
                error_text = stderr.decode("utf-8", "replace").strip()
                if len(error_text) > 1800:
                    error_text = error_text[-1800:]
                raise RuntimeError(error_text or f"FFmpeg завершился с кодом {return_code}")

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("FFmpeg не создал итоговый файл")

            output_size = output_path.stat().st_size
            upload_limit = (
                LOCAL_UPLOAD_SAFE_LIMIT if BOT_API_BASE else OFFICIAL_UPLOAD_LIMIT
            )

            if output_size > upload_limit:
                await safe_edit(
                    status,
                    (
                        f"Видео готово, но весит {fmt_bytes(output_size)}.\n\n"
                        f"Текущий режим Telegram не сможет его отправить. "
                        f"Для больших файлов используй Local Bot API. "
                        f"Его практический лимит - до 2 000 МБ на отправку."
                    ),
                )
                return

            await safe_edit(
                status,
                f"Рендер 100%\nОтправляю видео: {fmt_bytes(output_size)}",
            )

            video = FSInputFile(output_path, filename=output_path.name)
            caption = (
                f"{len(session.frames):,} кадров | "
                f"{session.width}x{session.height} | "
                f"{frame_duration} сек/кадр | "
                f"{fmt_duration(total_duration)}"
            ).replace(",", " ")

            try:
                await bot(
                    SendVideo(
                        chat_id=chat_id,
                        video=video,
                        width=session.width,
                        height=session.height,
                        duration=max(1, math.ceil(float(total_duration))),
                        supports_streaming=True,
                        caption=caption,
                    ),
                    request_timeout=3600,
                )
            except TelegramBadRequest:
                # Некоторые сочетания большого разрешения/контейнера Telegram
                # не любит как inline-video. Тогда отправляем тот же MP4 документом.
                await bot(
                    SendDocument(
                        chat_id=chat_id,
                        document=FSInputFile(output_path, filename=output_path.name),
                        caption=caption,
                    ),
                    request_timeout=3600,
                )

            session.last_used = time.monotonic()
            await safe_edit(
                status,
                "Готово. Кадры пока сохранены, можно сразу собрать еще одну скорость",
                reply_markup=done_keyboard(),
            )

    except asyncio.CancelledError:
        if proc is not None:
            await stop_process(proc)
        await safe_edit(status, "Рендер отменен")
        raise
    except Exception as exc:
        logging.exception("Render failed")
        text = str(exc).strip() or exc.__class__.__name__
        if len(text) > 2500:
            text = text[-2500:]
        await safe_edit(status, f"Рендер развалился:\n\n{text}")
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        if concat_path is not None:
            try:
                concat_path.unlink(missing_ok=True)
            except Exception:
                pass

        current = sessions.get(user_id)
        if current is session and current.render_task is asyncio.current_task():
            current.render_task = None


async def start_render(
    bot: Bot,
    chat_id: int,
    user_id: int,
    frame_duration: Decimal,
) -> bool:
    session = sessions.get(user_id)
    if session is None:
        await bot.send_message(chat_id, "Сначала пришли ZIP с кадрами")
        return False

    if session.render_task and not session.render_task.done():
        await bot.send_message(
            chat_id,
            "У тебя уже идет рендер. Если надо прибить его - /cancel",
        )
        return False

    session.waiting_custom_speed = False
    session.last_used = time.monotonic()
    task = asyncio.create_task(
        render_and_send(bot, chat_id, user_id, frame_duration)
    )
    session.render_task = task
    return True


async def delete_session(user_id: int) -> bool:
    session = sessions.pop(user_id, None)
    if session is None:
        return False

    if session.render_task and not session.render_task.done():
        session.render_task.cancel()
        try:
            await session.render_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    await asyncio.to_thread(safe_remove, session.root)
    return True


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Кидай ZIP с пронумерованными кадрами.\n\n"
        "Поддерживаются PNG, JPG, JPEG, WebP и BMP. "
        f"Максимум кадров сейчас: {MAX_FRAMES:,}.\n"
        "После загрузки выберешь длительность 1 кадра.".replace(",", " ")
    )


@router.message(Command("cancel"))
async def cancel_handler(message: Message) -> None:
    if not message.from_user:
        return

    session = sessions.get(message.from_user.id)
    if session is None or session.render_task is None or session.render_task.done():
        await message.answer("Сейчас ничего не рендерится")
        return

    session.render_task.cancel()
    await message.answer("Отмена отправлена FFmpeg")


@router.message(Command("delete"))
async def delete_handler(message: Message) -> None:
    if not message.from_user:
        return

    deleted = await delete_session(message.from_user.id)
    await message.answer("Кадры удалены" if deleted else "Удалять нечего")


@router.message(F.document)
async def document_handler(message: Message, bot: Bot) -> None:
    if not message.from_user or not message.document:
        return

    document = message.document
    filename = document.file_name or "frames.zip"

    if not filename.lower().endswith(".zip"):
        await message.answer("Нужен именно ZIP с кадрами")
        return

    if (
        not BOT_API_BASE
        and document.file_size
        and document.file_size > OFFICIAL_DOWNLOAD_LIMIT
    ):
        await message.answer(
            "Этот ZIP больше 20 МБ. Через обычный Telegram Bot API бот его "
            "скачать не сможет.\n\n"
            "Для больших архивов запусти комплект через docker compose - "
            "там используется Local Bot API без лимита на скачивание."
        )
        return

    user_id = message.from_user.id
    lock = get_lock(user_id)

    if lock.locked():
        await message.answer("Я уже обрабатываю твой предыдущий ZIP")
        return

    async with lock:
        status = await message.answer(
            f"Скачиваю ZIP"
            + (f" - {fmt_bytes(document.file_size)}" if document.file_size else "")
        )

        WORKDIR.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"framebot_{user_id}_", dir=WORKDIR))
        zip_path = root / "source.zip"

        try:
            await bot.download(
                document,
                destination=zip_path,
                timeout=3600,
                chunk_size=1024 * 1024,
            )

            await safe_edit(status, "Распаковываю и проверяю кадры")
            frames, width, height = await asyncio.to_thread(
                extract_and_validate,
                zip_path,
                root,
            )

            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass

            new_session = UserSession(
                root=root,
                frames_dir=root / "frames",
                frames=frames,
                width=width,
                height=height,
                source_name=filename,
            )

            old_session = sessions.get(user_id)
            sessions[user_id] = new_session

            if old_session is not None:
                if old_session.render_task and not old_session.render_task.done():
                    old_session.render_task.cancel()
                await asyncio.to_thread(safe_remove, old_session.root)

            await safe_edit(
                status,
                (
                    f"ZIP готов\n\n"
                    f"Кадров: {len(frames):,}\n"
                    f"Размер видео: {width}x{height}\n"
                    f"Архив: {filename}\n\n"
                    f"Выбери длительность 1 кадра"
                ).replace(",", " "),
                reply_markup=speed_keyboard(),
            )

        except Exception as exc:
            logging.exception("Archive processing failed")
            await asyncio.to_thread(safe_remove, root)
            text = str(exc).strip() or exc.__class__.__name__
            if len(text) > 2500:
                text = text[-2500:]
            await safe_edit(status, f"С ZIP что-то не так:\n\n{text}")


@router.callback_query(F.data.startswith("speed:"))
async def speed_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.from_user or not callback.data:
        return

    await callback.answer()
    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if session is None:
        if callback.message:
            await callback.message.answer("Архив уже удален. Пришли ZIP заново")
        return

    value = callback.data.split(":", 1)[1]

    if value == "custom":
        session.waiting_custom_speed = True
        session.last_used = time.monotonic()
        if callback.message:
            await callback.message.answer(
                "Напиши длительность 1 кадра.\n\n"
                "Примеры: 0.04, 0.07, 250ms, 1/25, 25fps"
            )
        return

    try:
        speed = parse_speed(value)
    except (InvalidOperation, ValueError):
        if callback.message:
            await callback.message.answer("Не понял скорость")
        return

    if callback.message:
        await start_render(
            bot,
            callback.message.chat.id,
            user_id,
            speed,
        )


@router.callback_query(F.data == "render:again")
async def render_again_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    if not callback.from_user or not callback.message:
        return

    session = sessions.get(callback.from_user.id)
    if session is None:
        await callback.message.answer("Кадры уже удалены. Пришли ZIP заново")
        return

    session.last_used = time.monotonic()
    await callback.message.answer(
        "Выбери новую длительность 1 кадра",
        reply_markup=speed_keyboard(),
    )


@router.callback_query(F.data == "session:delete")
async def delete_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    if not callback.from_user or not callback.message:
        return

    deleted = await delete_session(callback.from_user.id)
    await callback.message.answer("Кадры удалены" if deleted else "Удалять уже нечего")


@router.message(F.text)
async def text_handler(message: Message, bot: Bot) -> None:
    if not message.from_user or not message.text:
        return

    session = sessions.get(message.from_user.id)
    if session is None or not session.waiting_custom_speed:
        await message.answer("Пришли ZIP с кадрами")
        return

    try:
        speed = parse_speed(message.text)
    except (InvalidOperation, ValueError):
        await message.answer(
            "Не понял значение. Допустимо от 0.004 до 60 сек на кадр.\n"
            "Примеры: 0.04, 0.07, 250ms, 1/25, 25fps"
        )
        return

    await start_render(
        bot,
        message.chat.id,
        message.from_user.id,
        speed,
    )


async def cleanup_loop() -> None:
    ttl = SESSION_TTL_MINUTES * 60

    while True:
        await asyncio.sleep(300)
        now = time.monotonic()
        to_delete = []

        for user_id, session in list(sessions.items()):
            if session.render_task and not session.render_task.done():
                continue
            if now - session.last_used > ttl:
                to_delete.append(user_id)

        for user_id in to_delete:
            session = sessions.pop(user_id, None)
            if session is not None:
                await asyncio.to_thread(safe_remove, session.root)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if shutil.which(FFMPEG_BIN) is None:
        raise RuntimeError(
            f"FFmpeg не найден: {FFMPEG_BIN}. Установи ffmpeg или укажи FFMPEG_BIN"
        )

    WORKDIR.mkdir(parents=True, exist_ok=True)
    bot = build_bot()
    cleanup_task = asyncio.create_task(cleanup_loop())

    try:
        await dp.start_polling(bot)
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        for user_id in list(sessions):
            await delete_session(user_id)

        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
