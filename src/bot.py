import os
import logging
import uuid
import asyncio
from queue import Queue
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from concurrent.futures import ThreadPoolExecutor

from model_handler import generate_image_with_model

API_TOKEN = os.getenv('API_TOKEN')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_paths = {}
original_messages = {}

task_queue = Queue()
processing = False

executor = ThreadPoolExecutor(max_workers=4)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi!\nSend me any picture and choose a model to generate a new image. "
        "If your image is not within the size range of 64x64 to 512x512 pixels, it will automatically be adjusted before processing."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to image generation service! Here’s how to get started:\n"
        "1. Send me any picture.\n"
        "2. Choose a model from the list I provide.\n"
        "3. Your image will be placed in a processing queue. Images are processed one at a time, so there might be a wait, especially during busy periods. Processing can take a few minutes depending on the complexity and the current queue length.\n"
        "4. I'll resize your image if needed using bilinear interpolation, so it fits the required dimensions (64x64 to 512x512 pixels).\n"
        "5. After processing, I'll send you the newly generated image!\n"
        "Just upload your picture, and I'll guide you through the model selection and keep you updated on the progress."
    )


async def clear_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global task_queue, processing
    task_queue.queue.clear()
    processing = False
    await update.message.reply_text("The processing queue has been cleared.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    photo_path = photo_file.file_path

    image_id = str(uuid.uuid4())
    image_paths[image_id] = photo_path
    original_messages[image_id] = update.message

    keyboard = [
        [InlineKeyboardButton("Real-ESRGAN", callback_data=f'model_real_esrgan:{image_id}')],
        [InlineKeyboardButton("Ilia Real-ESRGAN", callback_data=f'model_ilia_real_esrgan:{image_id}')]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose the model to generate the image:", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    model_type, image_id = query.data.split(':')
    photo_path = image_paths.get(image_id)

    if photo_path:
        task = (query, model_type, image_id, photo_path)
        task_queue.put(task)
        await query.message.edit_text(f"Your image is in the queue. Position: {task_queue.qsize()}")

        global processing
        if not processing:
            processing = True
            asyncio.create_task(process_queue())
    else:
        await query.message.reply_text("Error: Image not found.")


async def process_queue():
    while not task_queue.empty():
        query, model_type, image_id, photo_path = task_queue.get()

        await query.message.edit_text("Your image is being processed. Please wait...")

        loop = asyncio.get_running_loop()
        generated_image = await loop.run_in_executor(executor, generate_image_with_model, model_type, photo_path)

        original_message = original_messages.get(image_id)
        if original_message:
            await original_message.reply_photo(photo=open(generated_image, 'rb'), quote=True)
            del original_messages[image_id]

        await query.message.delete()
        del image_paths[image_id]

    global processing
    processing = False


def main() -> None:
    application = Application.builder().token(API_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clearqueue", clear_queue))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()


if __name__ == '__main__':
    main()
