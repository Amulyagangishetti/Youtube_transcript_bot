from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import gradio as gr
from googleapiclient.discovery import build
from transformers import MarianMTModel, MarianTokenizer, pipeline
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi

API_KEY = "your_api_key"
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_video_id(url):
  
  parts = url.split('/')
  
  video_id_part = parts[-1].split('?')[0]
  return video_id_part

def get_transcript(video_id):
    """Fetch transcript for a given YouTube video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return  transcript
    except Exception as e:
        return f"Error fetching transcript: {e}"

def get_video_title(video_id):
  request=youtube.videos().list(part="snippet",id=video_id)
  response=request.execute()
  return response["items"][0]["snippet"]["title"]

os.environ["GROQ_API_KEY"]="your_groq_api_key"
llm=ChatGroq(model="llama3-8b-8192")
prompt=PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following video transcript:\n\n{transcript}"
)
def generate_summary(transcript):
  chain=LLMChain(llm=llm,prompt=prompt)
  summary=chain.run(transcript)
  return summary

def export_summary(summary, file_type):
    if file_type == "pdf":
        # Placeholder for PDF export logic
        file_path = "summary.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary)
        pdf.output(file_path)
        return file_path
    elif file_type == "txt":
        # Placeholder for TXT export logic
        file_path = "summary.txt"
        with open(file_path, "w") as file:
          file.write(summary)
        return file_path

def summarize_text(text):
    """Simplified summarization (can be replaced with a more sophisticated model)."""
    return text[:min(100, len(text))] + "..."


def summarize_with_timestamps(transcript):
    summaries = []
    for entry in transcript:
      if not isinstance(entry, dict) or 'start' not in entry or 'text' not in entry:
            continue  # Skip invalid entries
      timestamp = entry['start']
      summary = summarize_text(entry['text'])  # Placeholder for your summarize function
      summaries.append(f"{timestamp} - {summary}")
    return "\n".join(summaries)




def load_model(source_lang='en', target_lang='es'):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


model, tokenizer = load_model(source_lang='en', target_lang='es')

def translate_text(text, source_lang='en', target_lang='es', chunk_size=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    input_ids = inputs['input_ids']
    if input_ids.shape[1] > chunk_size:
        num_chunks = (input_ids.shape[1] // chunk_size) + 1
        chunks = [input_ids[:, i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    else:
        chunks = [input_ids]

    translated_chunks = []
    for chunk in chunks:
        translated = model.generate(chunk)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)

sentiment_analyzer = pipeline("sentiment-analysis")
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']


def process_video(url,target_lang='en'):
  video_id=get_video_id(url)
  transcript=get_transcript(video_id)
  title=get_video_title(video_id)
  if isinstance(transcript, str) and "Error" in transcript:
        return transcript, "Error"
  if not isinstance(transcript, list):
        return "Error: Invalid transcript format."
  timestamped_summary = summarize_with_timestamps(transcript)
  translated_summary = translate_text(timestamped_summary, source_lang='en', target_lang=target_lang)
  sentiment_label, sentiment_score = analyze_sentiment(translated_summary)
  summary=generate_summary(translated_summary)
  output = {
        'summary': summary,
        'sentiment': f"Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})",
        'translated_summary': translated_summary
  }
  return translated_summary,output['sentiment']

##Gradio UI code

with gr.Blocks() as demo:
    # Input components
    url_input = gr.Textbox(label="Enter your URL here")
    language_input = gr.Dropdown(label="Select language", choices=["en", "es", "fr", "de", "it"], value="en")


    # Button to trigger summarization
    summarize_button = gr.Button("Summarize")

    # Output components
    summary_output = gr.Textbox(label="Summarized Output")
    sentiment_output = gr.Textbox(label="Sentiment Analysis")

    # Button to export the summary
    export_button = gr.Button("Export Summary")
    export_file_output = gr.File(label="Download Summary")
    file_type_input = gr.Radio(choices=["pdf", "txt"], label="File Type")

    # Define the event listeners
    summarize_button.click(
        process_video,
        inputs=[url_input, language_input],
        outputs=[summary_output, sentiment_output]
    )

    export_button.click(
        fn=export_summary,
        inputs=[summary_output,file_type_input],
        outputs=export_file_output
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
