import ollama
from PIL import ImageGrab
import base64
import cv2
from cv2 import VideoCapture, imencode
from threading import Lock, Thread
import speech_recognition as sr
import whisper
import io
import re
from os import system

# CONSTS
wake_word = 'bobo'

# llm_model_name = 'llama3.1:70b'
llm_model_name = 'llama3.1'

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity, or any sentences like "Based on the image context and your user prompt"'
)
convo = [{'role': 'system', 'content': sys_msg}]


speech_process = None

# INIT
whisper_size = 'small.en'
whisper_model = whisper.load_model(whisper_size)

recognizer = sr.Recognizer ()
microphone = sr.Microphone()

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer), frame

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"]\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = ollama.chat(messages=function_convo, model='llama3.1')
    response = chat_completion['message']

    return response['content']

def llama_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'

    convo.append({'role': 'user', 'content': prompt})
    chat_completion = ollama.chat(model=llm_model_name, messages=convo)
    response = chat_completion['message']
    convo.append(response)

    return response['content']

def vision_prompt(prompt, photo):
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user.\nUSER PROMPT: {prompt}'
    )

    response = ollama.generate(model='llava', prompt=prompt, images=[photo])
    return response['response']

def take_screenshot():
    path = './tmp/screenshot.jpg'
    screenshot = ImageGrab.grab()

    buffered = io.BytesIO()
    rgb_screenshot = screenshot.convert('RGB')
    
    screenshot.save(buffered, format="PNG")
    rgb_screenshot.save(path, quality=15)

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def speak(text):
    # TODO 
    global speech_process
    ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ")
    clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
    system(f"say '{clean_text}'")
    # Start the speech synthesis process

    # speech_process = subprocess.Popen(["say", clean_text])

    # player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    # stream_start = False

    # with openai_client.audio.speech.with_streaming_response.create(
    #     model='tts-1',
    #     voice='onyx',
    #     response_format='pcm',
    #     input=text,
    # ) as response:
    #     silence_threshold = 0.01
    #     for chunk in response.iter_bytes(chunk_size=1024):
    #         if stream_start:
    #             player_stream.write(chunk)
    #         else:
    #             if max(chunk) > silence_threshold:
    #                 player_stream.write(chunk)
    #                 stream_start = True

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def audio_callback(recognizer, audio):
    print('Listening...\n')
    # prompt_audio_path = 'prompt.wav'
    # with open(prompt_audio_path, 'wb') as f:
    #     f.write(audio.get_wav_data())

    # prompt_text = wav_to_text(prompt_audio_path)
    # clean_prompt = extract_prompt(prompt_text, wake_word)

    # DEBUG
    prompt_audio_path = './tmp/prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    
    print('Processing Voice...\n')
    result = whisper_model.transcribe(prompt_audio_path, verbose=False, fp16=False)
    # result = whisper_model.transcribe(prompt_audio_path, verbose=False, language="en", fp16=False)
    # print(result["text"])

    clean_prompt = result["text"]
    # clean_prompt = recognizer.recognize_whisper(audio, model="base")

    if clean_prompt:
        print(f'> USER: {clean_prompt}\n')
        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print('DEBUG: Taking screenshot.\n')
            photo = take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo=photo)
        elif 'capture webcam' in call:
            print('DEBUG: Capturing webcam.\n')
            photo, frame = webcam_stream.read(encode=True)
             # DEBUG
            cv2.imwrite('./tmp/webcam.jpg', frame)

            visual_context = vision_prompt(prompt=clean_prompt, photo=photo)
        # elif 'extract clipboard' in call:
        #     print('Extracting clipboard text.')
        #     paste = get_clipboard_text()
        #     clean_prompt = f'{clean_prompt} \n\n CLIPBOARD CONTENT: {paste}'
        #     visual_context = None
        else:
            visual_context = None
       

        print(f'DEBUG visual_context: {visual_context}\n')
        response = llama_prompt(clean_prompt, visual_context)
        print(f'> ASSISTANT: {response}\n')
        speak(response)


# def start_listening():
#     with source as s:
#         r.adjust_for_ambient_noise(s)
#     # print('\nSay', wake_word, 'followed with your prompt. \n')
#     r.listen_in_background(source, callback)


def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*( [A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None
    

# RUN
webcam_stream = WebcamStream().start()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Webcam", 640, 360) 
    cv2.imshow("Webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
if speech_process is not None:
    speech_process.terminate()  # Terminates the speech process
    speech_process = None
