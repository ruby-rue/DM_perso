"""
Advanced Emotion Recognition Desktop App
Beautiful interface with your custom trained model
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
from tensorflow import keras
import threading
import time
from datetime import datetime
import json
import os

class AdvancedEmotionApp:
    def __init__(self, root, model_path='emotion_model_v2_best.h5'):
        self.root = root
        self.root.title("✨ Emotion Recognition Pro ✨")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Load custom model
        print("Loading your custom trained model...")
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from: {model_path}")
        except:
            messagebox.showerror("Error", f"Could not load model: {model_path}\nPlease train the model first!")
            self.root.destroy()
            return
        
        # Variables
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.emotions = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }
        
        self.emotion_history = []
        self.recording = False
        
        # Emotion emojis and colors
        self.emotion_emojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        
        self.emotion_colors = {
            'angry': '#FF6B6B',
            'disgust': '#51CF66',
            'fear': '#9775FA',
            'happy': '#FFD93D',
            'sad': '#74C0FC',
            'surprise': '#FF922B',
            'neutral': '#ADB5BD'
        }
        
        # Setup UI
        self.setup_modern_ui()
        
        # Start animations
        self.animation_cycle = 0
        self.animate()
    
    def setup_modern_ui(self):
        """Setup modern professional UI"""
        # Create custom style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom colors
        style.configure('Custom.TFrame', background='#1a1a2e')
        style.configure('Card.TFrame', background='#16213e', relief=tk.RAISED)
        
        # Header
        header = tk.Frame(self.root, bg='#0f3460', height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text='✨ Emotion Recognition Pro ✨',
            font=('Helvetica', 28, 'bold'),
            bg='#0f3460',
            fg='#E94560'
        )
        title.pack(pady=20)
        
        # Main content
        content = tk.Frame(self.root, bg='#1a1a2e')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Video
        left_panel = tk.Frame(content, bg='#16213e', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video frame
        video_container = tk.Frame(left_panel, bg='#000000')
        video_container.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_container, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Video overlay frame
        self.overlay_frame = tk.Frame(video_container, bg='white')
        self.overlay_frame.place(x=20, y=20)
        
        self.overlay_emoji = tk.Label(
            self.overlay_frame,
            text='😊',
            font=('Arial', 40),
            bg='white',
            fg='black'
        )
        self.overlay_emoji.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.overlay_text = tk.Label(
            self.overlay_frame,
            text='Happy',
            font=('Helvetica', 20, 'bold'),
            bg='white',
            fg='#0f3460'
        )
        self.overlay_text.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Controls
        controls = tk.Frame(left_panel, bg='#16213e')
        controls.pack(pady=15)
        
        btn_style = {
            'font': ('Helvetica', 12, 'bold'),
            'relief': tk.RAISED,
            'bd': 3,
            'padx': 25,
            'pady': 12,
            'cursor': 'hand2'
        }
        
        self.start_btn = tk.Button(
            controls,
            text='🎥 Start Camera',
            bg='#E94560',
            fg='white',
            activebackground='#c93750',
            command=self.toggle_camera,
            **btn_style
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = tk.Button(
            controls,
            text='📸 Capture',
            bg='#0f3460',
            fg='white',
            activebackground='#082644',
            command=self.capture_screenshot,
            **btn_style
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.record_btn = tk.Button(
            controls,
            text='⏺ Record',
            bg='#533483',
            fg='white',
            activebackground='#3e2661',
            command=self.toggle_recording,
            **btn_style
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_bar = tk.Frame(left_panel, bg='#0f3460', height=40)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_bar,
            text='Ready to start',
            font=('Helvetica', 11),
            bg='#0f3460',
            fg='white'
        )
        self.status_label.pack(pady=10)
        
        # Right panel - Emotions
        right_panel = tk.Frame(content, bg='#16213e', width=450, relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Panel header
        panel_header = tk.Label(
            right_panel,
            text='Emotion Analysis 💖',
            font=('Helvetica', 22, 'bold'),
            bg='#16213e',
            fg='#E94560'
        )
        panel_header.pack(pady=20)
        
        # Scrollable frame for emotions
        canvas = tk.Canvas(right_panel, bg='#16213e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#16213e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create emotion cards
        self.emotion_widgets = {}
        emotions_list = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        
        for emotion in emotions_list:
            card = self.create_modern_emotion_card(scrollable_frame, emotion)
            card.pack(fill=tk.X, pady=8, padx=10)
            self.emotion_widgets[emotion] = card
        
        # Statistics button
        stats_btn = tk.Button(
            right_panel,
            text='📊 View Statistics',
            font=('Helvetica', 12, 'bold'),
            bg='#0f3460',
            fg='white',
            activebackground='#082644',
            command=self.show_statistics,
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        stats_btn.pack(pady=15)
    
    def create_modern_emotion_card(self, parent, emotion):
        """Create a modern styled emotion card"""
        # Card container
        card = tk.Frame(parent, bg='#0f3460', relief=tk.RAISED, bd=3)
        
        # Inner content
        content = tk.Frame(card, bg='#0f3460')
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # Left: Emoji
        emoji_label = tk.Label(
            content,
            text=self.emotion_emojis[emotion],
            font=('Arial', 42),
            bg='#0f3460'
        )
        emoji_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # Middle: Info
        info = tk.Frame(content, bg='#0f3460')
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Name
        name = tk.Label(
            info,
            text=emotion.capitalize(),
            font=('Helvetica', 16, 'bold'),
            bg='#0f3460',
            fg='white',
            anchor='w'
        )
        name.pack(fill=tk.X)
        
        # Progress background
        progress_bg = tk.Frame(info, bg='#1a1a2e', height=18, relief=tk.SUNKEN, bd=1)
        progress_bg.pack(fill=tk.X, pady=(8, 0))
        progress_bg.pack_propagate(False)
        
        # Progress bar
        progress = tk.Frame(progress_bg, bg=self.emotion_colors[emotion])
        progress.place(x=0, y=0, relheight=1, width=0)
        
        # Right: Percentage
        percentage = tk.Label(
            content,
            text='0%',
            font=('Helvetica', 20, 'bold'),
            bg='#0f3460',
            fg=self.emotion_colors[emotion],
            width=7
        )
        percentage.pack(side=tk.RIGHT, padx=(15, 0))
        
        # Store references
        card.emoji = emoji_label
        card.name = name
        card.progress = progress
        card.percentage = percentage
        card.emotion = emotion
        card.content = content
        
        return card
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera and analysis"""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not access camera")
            return
        
        self.is_running = True
        self.start_btn.config(text='🛑 Stop Camera', bg='#c93750')
        self.status_label.config(text='✅ Analyzing emotions...')
        
        # Start threads
        threading.Thread(target=self.update_video, daemon=True).start()
        threading.Thread(target=self.analyze_emotions, daemon=True).start()
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        
        self.start_btn.config(text='🎥 Start Camera', bg='#E94560')
        self.status_label.config(text='Camera stopped')
        self.video_label.config(image='', bg='#000000')
    
    def update_video(self):
        """Update video feed"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            
            # Draw face rectangles
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Get dominant emotion color
                dominant = max(self.emotions, key=self.emotions.get)
                color_hex = self.emotion_colors[dominant]
                color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
            
            # Convert and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            time.sleep(0.03)
    
    def analyze_emotions(self):
        """Analyze emotions using custom model"""
        while self.is_running:
            if self.current_frame is not None:
                try:
                    gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        # Take first face
                        (x, y, w, h) = faces[0]
                        face = gray[y:y+h, x:x+w]
                        
                        # Preprocess
                        face_resized = cv2.resize(face, (48, 48))
                        face_normalized = face_resized.astype('float32') / 255.0
                        face_input = face_normalized.reshape(1, 48, 48, 1)
                        
                        # Predict
                        predictions = self.model.predict(face_input, verbose=0)[0]
                        
                        # Update emotions
                        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                        self.emotions = {
                            emotion_labels[i]: float(predictions[i] * 100)
                            for i in range(len(emotion_labels))
                        }
                        
                        # Save to history
                        if self.recording:
                            self.emotion_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'emotions': self.emotions.copy()
                            })
                        
                        # Update UI
                        self.root.after(0, self.update_emotion_display)
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            time.sleep(0.5)
    
    def update_emotion_display(self):
        """Update emotion display"""
        dominant = max(self.emotions, key=self.emotions.get)
        
        # Update overlay
        self.overlay_emoji.config(text=self.emotion_emojis[dominant])
        self.overlay_text.config(text=dominant.capitalize())
        self.overlay_frame.config(bg=self.emotion_colors[dominant])
        self.overlay_emoji.config(bg=self.emotion_colors[dominant])
        self.overlay_text.config(bg=self.emotion_colors[dominant], fg='white')
        
        # Update cards
        for emotion, value in self.emotions.items():
            card = self.emotion_widgets[emotion]
            
            # Update percentage
            card.percentage.config(text=f'{int(value)}%')
            
            # Update progress bar
            card.progress.place(relwidth=value/100)
            
            # Highlight dominant
            if emotion == dominant:
                card.config(bg=self.emotion_colors[emotion], bd=4)
                card.content.config(bg=self.emotion_colors[emotion])
                card.name.config(bg=self.emotion_colors[emotion], fg='white')
                card.percentage.config(bg=self.emotion_colors[emotion], fg='white')
                card.emoji.config(bg=self.emotion_colors[emotion])
            else:
                card.config(bg='#0f3460', bd=3)
                card.content.config(bg='#0f3460')
                card.name.config(bg='#0f3460', fg='white')
                card.percentage.config(bg='#0f3460')
                card.emoji.config(bg='#0f3460')
    
    def capture_screenshot(self):
        """Save screenshot"""
        if self.current_frame is not None:
            filename = f'emotion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f'✅ Saved: {filename}')
            messagebox.showinfo("Success", f"Screenshot saved as {filename}")
    
    def toggle_recording(self):
        """Toggle emotion recording"""
        self.recording = not self.recording
        
        if self.recording:
            self.record_btn.config(text='⏹ Stop Recording', bg='#c93750')
            self.emotion_history = []
            self.status_label.config(text='🔴 Recording emotions...')
        else:
            self.record_btn.config(text='⏺ Record', bg='#533483')
            
            if self.emotion_history:
                # Save history
                filename = f'emotions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(filename, 'w') as f:
                    json.dump(self.emotion_history, f, indent=2)
                
                messagebox.showinfo("Success", f"Emotion history saved as {filename}")
                self.status_label.config(text=f'✅ Saved: {filename}')
    
    def show_statistics(self):
        """Show emotion statistics"""
        if not self.emotion_history:
            messagebox.showinfo("No Data", "Start recording to collect emotion data!")
            return
        
        # Calculate statistics
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Emotion Statistics")
        stats_window.geometry("600x500")
        stats_window.configure(bg='#1a1a2e')
        
        # Title
        title = tk.Label(
            stats_window,
            text="📊 Emotion Statistics",
            font=('Helvetica', 20, 'bold'),
            bg='#1a1a2e',
            fg='#E94560'
        )
        title.pack(pady=20)
        
        # Stats text
        stats_text = tk.Text(
            stats_window,
            font=('Courier', 12),
            bg='#0f3460',
            fg='white',
            wrap=tk.WORD,
            padx=20,
            pady=20
        )
        stats_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Calculate averages
        avg_emotions = {emotion: 0 for emotion in self.emotions}
        for entry in self.emotion_history:
            for emotion, value in entry['emotions'].items():
                avg_emotions[emotion] += value
        
        for emotion in avg_emotions:
            avg_emotions[emotion] /= len(self.emotion_history)
        
        # Display stats
        stats_text.insert('end', f"Total Recordings: {len(self.emotion_history)}\n\n")
        stats_text.insert('end', "Average Emotions:\n")
        stats_text.insert('end', "-" * 40 + "\n")
        
        for emotion, value in sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True):
            emoji = self.emotion_emojis[emotion]
            stats_text.insert('end', f"{emoji} {emotion.capitalize():12} {value:6.2f}%\n")
        
        stats_text.config(state=tk.DISABLED)
    
    def animate(self):
        """Subtle animations"""
        self.animation_cycle += 1
        
        # Pulse overlay
        if self.animation_cycle % 30 < 15:
            self.overlay_frame.config(relief=tk.RAISED, bd=3)
        else:
            self.overlay_frame.config(relief=tk.RAISED, bd=2)
        
        self.root.after(100, self.animate)
    
    def on_closing(self):
        """Cleanup"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

def main():
    """Main function"""
    print("\n" + "="*70)
    print("✨ ADVANCED EMOTION RECOGNITION APP ✨")
    print("="*70)
    
    # Check for model
    model_files = [
        'emotion_model_v2_best.h5',
        'emotion_model_v1_best.h5',
        'emotion_model_v3_best.h5',
        'emotion_model_trained.h5'
    ]
    
    model_path = None
    for model in model_files:
        if os.path.exists(model):
            model_path = model
            break
    
    if not model_path:
        print("\n❌ No trained model found!")
        print("\nPlease train a model first:")
        print("  python train_custom_model.py")
        print("\nOr specify model path when creating app:")
        print("  app = AdvancedEmotionApp(root, 'your_model.h5')")
        return
    
    print(f"\nUsing model: {model_path}")
    print("Loading app...")
    
    root = tk.Tk()
    app = AdvancedEmotionApp(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()