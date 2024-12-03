import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory,session, flash
import pymongo
from bson import ObjectId
import base64
import gridfs
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from dotenv import load_dotenv
from flask_mail import Mail, Message
from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import io
from tensorflow.keras.models import  load_model
import pandas as pd
from pydub import AudioSegment
import seaborn as sns
import re
import whisperx
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

 # Secure and randomly generated secret key

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI')

# MONGO_DATABASE = 'agent_db'
MONGO_DATABASE = 'behind_the_voice'


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]
# users_collection = db.profile
agent_collection = db.agent_profile
admin_collection = db.admin
fs = gridfs.GridFS(db)

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load emotions dictionary

emotions = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}
emo_list = list(emotions.values())

# Diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token='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'
)



# Load the model
model= 'D:/Downloads/Menna/CNN_combined_model.h5'
model = load_model(model)

def extract_features(data, sr):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_predict_feat(audio_path):
    d, sr = librosa.load(audio_path)
    res = extract_features(d, sr)
    target_length = 162
    if len(res) < target_length:
        res = np.pad(res, (0, target_length - len(res)))
    elif len(res) > target_length:
        res = res[:target_length]
    result = np.reshape(res, newshape=(target_length, 1))
    return result


def process_audio(audio_path):
    # Perform speaker diarization
    diarization = pipeline(audio_path, num_speakers=2)
    audio = AudioSegment.from_wav(audio_path)
    segments = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        stop_time = turn.end
        start_ms = int(start_time * 1000)
        stop_ms = int(stop_time * 1000)
        segment = audio[start_ms:stop_ms]
        speaker_index = int(speaker.split("_")[-1])

        if speaker_index not in segments:
            segments[speaker_index] = []
        segments[speaker_index].append(segment)

    for speaker_index, speaker_segments in segments.items():
        # Export individual segments
        for i, segment in enumerate(speaker_segments):
            speaker_filename = f"{app.config['UPLOAD_FOLDER']}/speaker_{speaker_index}_{i}.wav"
            segment.export(speaker_filename, format="wav")

        # Concatenate the segments for the current speaker
        concatenated_segment = sum(speaker_segments)
        concatenated_filename = f"{app.config['UPLOAD_FOLDER']}/concatenated_speaker_{speaker_index}.wav"
        concatenated_segment.export(concatenated_filename, format="wav")

    with open(f"{app.config['UPLOAD_FOLDER']}/sample.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

    df = rttm_to_dataframe(f"{app.config['UPLOAD_FOLDER']}/sample.rttm")
    df = df.astype({'Start Time': 'float', 'Duration': 'float'})
    df['End Time'] = df['Start Time'] + df['Duration']
    df['emotion'] = None

    # Drop unwanted columns
    df = df.drop(["Type", "File ID", "Channel"], axis=1)

    # Add emotions to DataFrame
    count0, count1 = 0, 0
    for i in range(len(df)):
        speaker_id = df.loc[i, 'Speaker']
        if speaker_id.endswith('00'):
            preprocessed_audio = get_predict_feat(f"{app.config['UPLOAD_FOLDER']}/speaker_0_{count0}.wav")

            prediction = model.predict(np.expand_dims(preprocessed_audio, axis=0))
            max_index = np.argmax(prediction)
            emotion = emo_list[max_index]
            df.at[i, 'emotion'] = emotion
            count0 += 1
        if speaker_id.endswith('01'):
            preprocessed_audio = get_predict_feat(f"{app.config['UPLOAD_FOLDER']}/speaker_1_{count1}.wav")

            prediction = model.predict(np.expand_dims(preprocessed_audio, axis=0))
            max_index = np.argmax(prediction)
            emotion = emo_list[max_index]
            df.at[i, 'emotion'] = emotion
            count1 += 1

    # Split into two dataframes based on Speaker
    df_speaker_01 = df[df["Speaker"] == "SPEAKER_01"].reset_index(drop=True)
    df_speaker_00 = df[df["Speaker"] == "SPEAKER_00"].reset_index(drop=True)

    return df, df_speaker_00, df_speaker_01


def rttm_to_dataframe(rttm_file_path):
    columns = ["Type", "File ID", "Channel", "Start Time", "Duration", "Orthography", "Confidence", "Speaker", 'x', 'y']
    with open(rttm_file_path, "r") as rttm_file:
        lines = rttm_file.readlines()
    data = []
    for line in lines:
        line = line.strip().split()
        data.append(line)
    df = pd.DataFrame(data, columns=columns)
    df = df.drop(['x', 'y', "Orthography", "Confidence"], axis=1)
    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        jobtitle = request.form['jobtitle']
        phone = request.form['phone']
        address = request.form['address']
        email = request.form['email']
        password = request.form['password']
        gender = request.form['gender']
        birthday = request.form['birthday']
        profile_image = request.files['profile_image']

        if profile_image and allowed_file(profile_image.filename):
            filename = secure_filename(profile_image.filename)
            profile_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            filename = None

        hashed_password = generate_password_hash(password)

        user_data = {
            'name': name,
            'jobtitle': jobtitle,
            'phone': phone,
            'address': address,
            'email': email,
            'password': hashed_password,
            'gender': gender,
            'birthday': birthday,
            'profile_image': filename
        }

        agent_collection.insert_one(user_data)
           # Send a welcome email to the newly registered agent
        recipient_email = request.form['email']
        subject = 'Welcome to Our Platform'
        message_content = f'Hi {name},\n\nWelcome to our platform Behind the voice! You have successfully registered as an agent.'
        msg = Message(subject, sender='hallaorders581@gmail.com', recipients=[recipient_email])
        msg.body = message_content
        mail.send(msg)

        # Automatically log the user in after signup
        session['name'] = name
        flash('Signup successful! Welcome to your profile.')

        return redirect(url_for('profile', name=name))

    return render_template('signup.html')



@app.route('/profile/<name>')
def profile(name):
    user_data = agent_collection.find_one({'name': name})
    if not user_data:
        flash('User not found!')
        return redirect(url_for('login'))
    return render_template('profile.html', user=user_data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        user = agent_collection.find_one({'name': name})
        
        if user and check_password_hash(user['password'], password):
            session['name'] = name
            flash('Login successful!')
            return redirect(url_for('profile', name=name))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('login'))
    
    return render_template('login.html')



@app.route('/logout')
def logout():
    # Clear the user's session
    session.clear()
    
    # Flash a message to indicate successful logout
    flash('You have been logged out.')
    
    # Redirect the user to the index page
    return redirect(url_for('index'))


@app.route('/admin_signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if admin_collection.find_one({'username': username}):
            flash('Username already exists!')
            return redirect(url_for('admin_signup'))
        
        hashed_password = generate_password_hash(password)
        admin_collection.insert_one({'username': username, 'password': hashed_password})
        # Automatically log the user in after signup
        session['username'] = username
        flash('Signup successful! Welcome to your profile.')
        return redirect(url_for('get_agents'))
    
    return render_template('admin_signup.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = admin_collection.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!')
            return redirect(url_for('get_agents'))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('admin_login'))
    
    return render_template('admin_login.html')

# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'hallaorders581@gmail.com'  # Replace with sender mail
app.config['MAIL_PASSWORD'] = 'njuk sicw qmkx dxrx'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

@app.route('/send_mail', methods=['POST', 'GET'])
def send_mail():
    if request.method == 'POST':
       
        recipient_email = 'mennaelghanam90@gmail.com'  # Replace with the recipient email address
        subject = 'Hello'
        message_content = request.form['message']

        msg = Message(subject, sender='hallaorders581@gmail.com', recipients=[recipient_email])
        msg.body = message_content
        mail.send(msg)
        flash('Email sent successfully!')
        return redirect(url_for('send_mail'))
        # return 'Email sent successfully!'
    return render_template('send_mail.html')

@app.route('/agents', methods=['GET'])
def get_agents():
    search_query = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    query = {}
    if search_query:
        query = {"$or": [
            {"agent_name": {"$regex": search_query, "$options": "i"}},
            {"agent_phone": {"$regex": search_query, "$options": "i"}},
            {"agent_email": {"$regex": search_query, "$options": "i"}}
        ]}

    total_items = db.calls.count_documents(query)
    total_pages = (total_items + per_page - 1) // per_page

    agents = list(db.calls.find(query).skip((page - 1) * per_page).limit(per_page))

    for agent in agents:
        audio_id = agent.get('audio_id')
        audio_data = fs.get(audio_id).read()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        agent['audio_b64'] = audio_b64

        file_info = db.fs.files.find_one({'_id': audio_id})
        if file_info:
            upload_date_time = file_info.get('uploadDate')
            formatted_date_time = datetime.strftime(upload_date_time, "%d/%m/%Y %H:%M:%S")
            agent['upload_date_time'] = formatted_date_time
        else:
            agent['upload_date_time'] = "N/A"

    return render_template('agents.html', agents=agents, page=page, per_page=per_page, total_pages=total_pages, search_query=search_query)




# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No audio file selected!')
#             return redirect(request.url)

#         file = request.files['file']

#         if file.filename == '':
#             flash('No selected file!')
#             return redirect(request.url)

#         if file:
#             filename = secure_filename(file.filename)

#             # Optional: Save audio file locally for testing (adjust path)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             print(file_path)
#             file.seek(0)

#             # Store audio file in GridFS
#             audio_data = file.read()  # Read audio data
#             audio_id = fs.put(audio_data, filename=filename)  # Store in GridFS

#             # Collect agent information
#             agent_name = request.form['agent_name']
#             agent_id = request.form['agent_id']

#             # Create a dictionary to store agent data and audio ID
#             agent_data = {
#                 "agent_name": agent_name,
#                 "agent_id": agent_id,
#                 "audio_id": audio_id,
#             }

#             # Insert agent data into MongoDB (assuming a collection named 'Agent')
#             db.calls.insert_one(agent_data)
            
#             # Process the audio and get the DataFrames
#             # df, df_speaker_00, df_speaker_01 = process_audio(file_path)

#             # Convert DataFrames to HTML tables
#             # df_html = df.to_html(classes='data')
#             # df_speaker_00_html = df_speaker_00.to_html(classes='data')
#             # df_speaker_01_html = df_speaker_01.to_html(classes='data')

#             # Flash success message
#             flash('Recording uploaded and processed successfully!')
           

#             # Render the template with the DataFrame tables
           

#             return render_template('agents.html' )
                                   
#     else:
#         return render_template('upload.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No audio file selected!')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file!')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)

            # Optional: Save audio file locally for testing (adjust path)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            file.seek(0)

            # Store audio file in GridFS
            audio_data = file.read()  # Read audio data
            audio_id = fs.put(audio_data, filename=filename)  # Store in GridFS

            # Collect agent information
            agent_name = request.form['agent_name']
            agent_id = request.form['agent_id']

            # Create a dictionary to store agent data and audio ID
            agent_data = {
                "agent_name": agent_name,
                "agent_id": agent_id,
                "audio_id": audio_id,
            }

            # Insert agent data into MongoDB (assuming a collection named 'calls')
            db.calls.insert_one(agent_data)
            
            # Flash success message
            flash('Recording uploaded and processed successfully!')
            
            # Redirect to the get_agents route (assuming this route displays the agents)
            return redirect(url_for('get_agents'))

    else:
        return render_template('upload.html')


# Whisper model setup
device = "cpu"
batch_size = 8  # Reduce if low on GPU memory
compute_type = "int8"
language = 'en'

whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)





@app.route('/transcribe', methods=['POST', 'GET'])
def transcribe():
    try:
        # Get the audio_id from URL parameters
        audio_id = request.args.get('id')
        audio_data = fs.get(ObjectId(audio_id))
        if audio_data is None:
            return jsonify({'error': 'Audio not found'}), 404

        audio_binary = audio_data.read()
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_data.filename)
        with open(audio_path, 'wb') as f:
            f.write(audio_binary)

        # Transcribe concatenated audio
        concatenated_transcriptions = {}
        for speaker_index in [0, 1]:  # Assuming 2 speakers
            concatenated_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"concatenated_speaker_{speaker_index}.wav")
            concatenated_transcriptions[speaker_index] = transcribe_file(concatenated_filename)

        return render_template('transcribe.html', transcriptions=concatenated_transcriptions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def rttm_to_dataframe(rttm_file_path):
    columns = ["Type", "File ID", "Channel", "Start Time", "Duration", "Orthography", "Confidence", "Speaker", 'x', 'y']
    with open(rttm_file_path, "r") as rttm_file:
        lines = rttm_file.readlines()
    data = []
    for line in lines:
        data.append(line.strip().split())
    df = pd.DataFrame(data, columns=columns)
    df = df.drop(['x', 'y', "Orthography", "Confidence"], axis=1)
    return df

def transcribe_file(audio_file):
    if not os.path.exists(audio_file):
        print(f"File {audio_file} not found!")
        return []
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, batch_size=batch_size)
    return result["segments"]





@app.route('/predict', methods=['GET'])
def predict():
    try:
        audio_id = request.args.get('id')
        audio_data = fs.get(ObjectId(audio_id))
        if audio_data is None:
            return jsonify({'error': 'Audio not found'}), 404

        audio_binary = audio_data.read()
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_data.filename)
        with open(audio_path, 'wb') as f:
            f.write(audio_binary)

        df, df_speaker_00, df_speaker_01 = process_audio(audio_path)

        # Plot for Speaker 01
        fig_speaker_01 = px.scatter(df_speaker_01, x='Start Time', y='End Time', color='emotion',
                                    title='End Time vs Start Time with Emotion (Speaker 01)')
        plot_url_speaker_01 = pio.to_html(fig_speaker_01, full_html=False)

        # Plot for Speaker 00
        fig_speaker_00 = px.scatter(df_speaker_00, x='Start Time', y='End Time', color='emotion',
                                    title='End Time vs Start Time with Emotion (Speaker 00)')
        plot_url_speaker_00 = pio.to_html(fig_speaker_00, full_html=False)

        # Bar Plot for Speaker 01
        bar_plot_speaker_01 = go.Figure()
        bar_plot_speaker_01.add_trace(go.Bar(x=df_speaker_01['emotion'], y=df_speaker_01['Duration']))
        bar_plot_speaker_01.update_layout(title='Emotion Distribution for Speaker 01')

        bar_plot_speaker_01_div = bar_plot_speaker_01.to_html(full_html=False)

        # Bar Plot for Speaker 00
        bar_plot_speaker_00 = go.Figure()
        bar_plot_speaker_00.add_trace(go.Bar(x=df_speaker_00['emotion'], y=df_speaker_00['Duration']))
        bar_plot_speaker_00.update_layout(title='Emotion Distribution for Speaker 00')

        bar_plot_speaker_00_div = bar_plot_speaker_00.to_html(full_html=False)


        return render_template("results.html", 
                               tables=df.to_html(classes='data'), 
                               df_speaker_00=df_speaker_00.to_html(classes='data'), 
                               df_speaker_01=df_speaker_01.to_html(classes='data'), 
                               plot_url_speaker_01=plot_url_speaker_01,
                               plot_url_speaker_00=plot_url_speaker_00,
                               bar_plot_speaker_01=bar_plot_speaker_01_div,
                               bar_plot_speaker_00=bar_plot_speaker_00_div)
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)









