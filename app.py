import streamlit as st
import sqlite3
import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from fpdf import FPDF
pdf = FPDF()
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
import google.generativeai as genai

# Configure Gemini using your environment variable
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="🌱 AI Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PATHS ---------------- #

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "database" / "database.db"
MODEL_PATH = BASE_DIR / "model" / "Trained_Model_v1.2.keras"
UPLOAD_FOLDER = BASE_DIR / "uploads"

os.makedirs(UPLOAD_FOLDER / "profiles", exist_ok=True)
os.makedirs(UPLOAD_FOLDER / "leaves", exist_ok=True)

# ---------------- YOUR 44 CLASS NAMES ---------------- #
CLASS_NAMES = [
    "apple__apple_scab", "apple__black_rot", "apple__cedar_apple_rust", "apple__healthy",
    "cassava__bacterial_blight_cbb", "cassava__brown_streak_disease_cbsd", "cassava__green_mottle_cgm", 
    "cassava__healthy", "cassava__mosaic_disease_cmd",
    "cherry_including_sour__healthy", "cherry_including_sour__powdery_mildew",
    "corn_maize__cercospora_leaf_spot_gray_leaf_spot", "corn_maize__common_rust", "corn_maize__healthy", 
    "corn_maize__northern_leaf_blight",
    "grape__black_rot", "grape__esca_black_measles", "grape__healthy", "grape__leaf_blight_isariopsis_leaf_spot",
    "orange__haunglongbing_citrus_greening",
    "peach__bacterial_spot", "peach__healthy",
    "pepper_bell__bacterial_spot", "pepper_bell__healthy",
    "potato__early_blight", "potato__healthy", "potato__late_blight",
    "rice__brownspot", "rice__healthy", "rice__hispa", "rice__leafblast",
    "squash__powdery_mildew",
    "strawberry__healthy", "strawberry__leaf_scorch",
    "tomato__bacterial_spot", "tomato__early_blight", "tomato__healthy", "tomato__late_blight",
    "tomato__leaf_mold", "tomato__septoria_leaf_spot", "tomato__spider_mites_two-spotted_spider_mite",
    "tomato__target_spot", "tomato__tomato_mosaic_virus", "tomato__tomato_yellow_leaf_curl_virus"
]

# ---------------- DATABASE ---------------- #
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

conn = get_db_connection()
c = conn.cursor()

def init_db():
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            profile_pic TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            leaf_image TEXT NOT NULL,
            disease TEXT,
            confidence REAL,
            top3_alternatives TEXT,
            plant_type TEXT DEFAULT 'Unknown',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    c.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in c.fetchall()]
    if 'top3_alternatives' not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN top3_alternatives TEXT")
    
    conn.commit()

init_db()

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_ai_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape[1:3]
    return model, input_shape

model, MODEL_INPUT_SHAPE = load_ai_model()

# ---------------- PREPROCESS ---------------- #
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=MODEL_INPUT_SHAPE)
    img_array = image.img_to_array(img)
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_top_predictions(probs):
    top_indices = np.argsort(probs[0])[::-1][:3]
    top_probs = probs[0][top_indices] * 100
    top_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i+1}" for i in top_indices]
    return list(zip(top_names, [round(p, 1) for p in top_probs]))

def get_ai_treatment_tips(disease_name):
    """Fetches expert care advice from Gemini."""
    prompt = f"""
    You are an expert plant pathologist. 
    The AI has detected the following disease: {disease_name}.
    Provide a concise, professional, and actionable treatment plan for this plant disease.
    Include 3 key sections: Prevention, Immediate Treatment, and Best Practices.
    Keep it under 150 words.
    """
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error fetching AI advice: {e}")
        return "Advice currently unavailable. Please ensure proper plant hygiene, remove infected leaves, and consult a local agricultural extension office."

# ---------------- AUTH HELPERS ---------------- #
def login_user(email, password):
    email = email.lower().strip()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    return c.fetchone()

# ---------------- GLOBAL STATE ---------------- #
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.main, .stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    font-family: 'Poppins', sans-serif;
}
.auth-form{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius:20px;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
    max-width:650px;
    margin:auto;
    padding:3rem;
}
.pred-card {
    background: linear-gradient(135deg, #00d4aa, #00b894);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
}
.alt-card {
    background: rgba(255,255,255,0.1);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HOME PAGE ---------------- #
if st.session_state.page == 'home':
    st.markdown('<div class="landing-hero">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="text-align:center;">
            <h1 style="font-size:3.5rem; font-weight:700; background: linear-gradient(135deg,#00d4aa,#00b894,#00cec9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:1rem;">
            🌱 AI Plant Disease Detection
            </h1>
            <p style="font-size:1.4rem; color:#ffffff; font-weight:500; margin-bottom:2rem;">
                Instant diagnosis of <span style="color:#00d4aa;font-weight:700;">40+ plant diseases</span>
                across <span style="color:#ffeaa7;">10+ crops</span> with
                <span style="color:#55efc4;font-weight:700;">88% accuracy</span>
                in under <span style="color:#74b9ff;">30 seconds</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:left; font-size:1.1rem; color:#dfe6e9; line-height:1.6; margin-left: 20px;">
            <p><span style="color:#00d4aa;font-weight:700;">🔬 Powered by Deep Learning:</span> Advanced CNN models trained on <span style="color:#ffeaa7;">50,000+ images</span></p>
            <p><span style="color:#55efc4;font-weight:700;">🌾 Crops Covered:</span> Rice, Tomato, Potato, Apple, Corn, Pepper, Grape, Peach & more</p>
            <p><span style="color:#74b9ff;font-weight:700;">⚡ Real-time Results:</span> Mobile-optimized for instant field diagnosis</p>
            <p><span style="color:#fd79a8;font-weight:700;">📱 Works Offline:</span> Once trained, runs locally on any device</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align:center; margin-top: 2rem;">
            <div style="font-size:5rem; margin-bottom:1rem;">🎯</div>
            <div style="background: linear-gradient(135deg,#00d4aa,#00b894); color:white; padding:1.5rem; border-radius:20px; box-shadow:0 15px 30px rgba(0,212,170,0.3);">
                <div style="font-size:3.5rem; font-weight:800; margin-bottom:0.5rem;">88%</div>
                <h3 style="margin:0; font-size:1.2rem; letter-spacing:1px;">ACCURACY</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()
    with col_btn2:
        if st.button("👁️ Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()

    st.write("---")
    st.markdown("## 🚀 Key Features")
    f_col1, f_col2, f_col3 = st.columns(3)
    features = [
        ("⚡ Instant Detection", "Upload a leaf image and get prediction in <span style='color:#00b894;'>under 2 seconds</span>.", "#00d4aa"),
        ("🌾 Multiple Crops", "Detect diseases in <span style='color:#e17055;'>Rice</span>, <span style='color:#d63031;'>Tomato</span>, Potato and more.", "#6c5ce7"),
        ("📊 AI Accuracy", "Our CNN model provides up to <span style='color:#00b894;'>88% prediction accuracy</span>.", "#e84393")
    ]
    for i, (title, desc, color) in enumerate(features):
        cols = [f_col1, f_col2, f_col3]
        cols[i].markdown(f"""
        <div style="padding:1.5rem; border-radius:15px; border:1px solid #444; background:#1e1e1e; height:180px;">
            <h3 style="color:{color};">{title}</h3>
            <p style="color:#ecf0f1;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌿 Supported Crops")
    crops = ["🌾 Rice", "🍅 Tomato", "🥔 Potato", "🍎 Apple", "🌽 Corn", "🍇 Grape", "🍑 Peach", "🌶 Pepper", "🍓 Strawberry", "🍊 Orange"]
    crop_cols = st.columns(5)
    for i, crop in enumerate(crops):
        crop_cols[i % 5].info(crop)

    st.markdown("""
    <hr><div style="text-align:center; color:#888; font-size:12px; padding:20px;">
    © 2026 AI Plant Disease Detection | Built with ❤️ using Streamlit & Deep Learning
    </div>
    """, unsafe_allow_html=True)


#-------------SIGNUP-PAGE-----------------------#

elif st.session_state.page == 'signup':
    st.markdown("""
        <style>
        .signup-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            max-width: 600px;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style='text-align: center; color: #00d4aa; margin-bottom: 0;'>Join the Community</h2>
        <p style='text-align: center; color: #dfe6e9; margin-bottom: 30px;'>Start detecting plant diseases today</p>
    """, unsafe_allow_html=True)

    # First and Last Name
    col_name1, col_name2 = st.columns(2)
    with col_name1:
        f_name = st.text_input("First Name", placeholder="John")
    with col_name2:
        l_name = st.text_input("Last Name", placeholder="Doe")

    email = st.text_input("Email Address", placeholder="example@email.com")
    
    # Password and Confirm Password
    col_pwd1, col_pwd2 = st.columns(2)
    with col_pwd1:
        pwd = st.text_input("Create Password", type="password", placeholder="••••••••")
    with col_pwd2:
        confirm_pwd = st.text_input("Confirm Password", type="password", placeholder="••••••••")
    
    # Profile Picture Section (Optional)
    st.markdown("<p style='color: #00d4aa; font-weight: 600; margin-top: 15px;'>Profile Picture (Optional)</p>", unsafe_allow_html=True)
    profile_pic = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Registration Logic
    if st.button("✨ Create My Account", use_container_width=True):
        if not (f_name and l_name and email and pwd and confirm_pwd):
            st.error("⚠️ Please fill in all fields.")
        elif pwd != confirm_pwd:
            st.error("❌ Passwords do not match. Please check again.")
        else:
            try:
                # Handle profile picture path
                pic_path = ""
                if profile_pic:
                    pic_path = f"{UPLOAD_FOLDER}/profiles/{email.lower().strip()}_{profile_pic.name}"
                    with open(pic_path, "wb") as f:
                        f.write(profile_pic.getbuffer())
                
                c.execute("""
                    INSERT INTO users(first_name, last_name, email, password, profile_pic) 
                    VALUES(?,?,?,?,?)
                """, (f_name, l_name, email.lower().strip(), pwd, pic_path))
                conn.commit()
                
                st.balloons()
                st.success("🎉 Welcome aboard! Account created successfully.")
                st.info("Directing you to Login...")
                st.session_state.page = 'login'
                st.rerun()
                
            except sqlite3.IntegrityError:
                st.error("⚠️ This email is already registered. Try logging in!")

    # Navigation Links (Fixed buttons by removing 'variant')
    st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Already a member? Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    with c2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LOGIN PAGE ---------------- #

elif st.session_state.page == 'login':
    # Container for the form
    st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style='text-align: center; color: #00d4aa; margin-bottom: 5px;'>Welcome Back!</h2>
        <p style='text-align: center; color: #dfe6e9; margin-bottom: 25px;'>Please login to your account</p>
    """, unsafe_allow_html=True)

    # Input Fields
    email = st.text_input("Email Address", placeholder="example@email.com")
    password = st.text_input("Password", type="password", placeholder="••••••••")

    # Login Logic
    if st.button("🚀 Login", use_container_width=True):
        email_norm = email.lower().strip()
        user = login_user(email_norm, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user = {"name": f"{user[1]} {user[2]}", "email": user[3]}
            st.success("✅ Login successful! Redirecting...")
            st.session_state.page = 'dashboard'
            st.rerun()
        else:
            st.error("❌ Invalid email or password. Please try again.")

    # Navigation Links
    st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✨ Don't have an account? Sign Up", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()
    with col_b:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- DASHBOARD ---------------- #

elif st.session_state.page == 'dashboard':
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()

    # Sidebar
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.user['name'].split()[0]}")
    page = st.sidebar.radio("Navigate", ["🌿 Detect Disease", "📜 Prediction History", "👤 Account Profile", "⚙️ Settings","🚪 Logout"])
    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Ensure your leaf images are well-lit for better accuracy.")

    if page == "🌿 Detect Disease":
        st.title("🌿 Leaves Disease Detection")
        img = st.file_uploader("Upload leaf image", type=['jpg', 'png', 'jpeg'])
        
        if img:
            st.image(img, width=300)
            if st.button("🚀 Predict Disease ", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = f"{UPLOAD_FOLDER}/leaves/leaf_{ts}_{img.name}"
                    with open(img_path, "wb") as f: f.write(img.getbuffer())
                    
                    img_array = preprocess_image(img_path)
                    preds = model.predict(img_array)
                    top3 = get_top_predictions(preds)
                    
                    # Results Section
                    st.subheader("Analysis Results")
                    r_col1, r_col2 = st.columns([1, 2])
                    with r_col1:
                        st.metric("Top Disease", top3[0][0])
                        st.metric("Confidence", f"{float(top3[0][1]):.2f}%")
                    with r_col2:
                        st.write("**Top 3 Probabilities:**")
                        for name, conf in top3:
                            val = float(conf)
                            st.write(f"**{name}**: `{val:.2f}%`")
                            st.progress(val / 100)
                    
                    # ... (Existing code above remains exactly the same)
                    
                    # 3. Save to Database
                    top3_text = "; ".join([f"{n} ({float(c):.2f}%)" for n, c in top3])
                    c.execute("INSERT INTO predictions(user_email, leaf_image, disease, confidence, top3_alternatives, created_at) VALUES(?,?,?,?,?,?)",
                              (st.session_state.user["email"], img_path, top3[0][0], float(top3[0][1]), top3_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    st.success("✅ Analysis saved.")

                    

                    
                    
                    # Generate AI Advice
                    pdf.add_page()
                    st.markdown("---")
                    st.subheader("💡 AI Recommended Treatment")

                    with st.spinner("Gemini is consulting the expert database..."):
                        treatment_advice = get_ai_treatment_tips(top3[0][0])
                        # CUSTOM STYLE BOX
                        st.markdown(f"""
                        <div style="background-color: #1e2a2a; border-left: 5px solid #00d4aa; 
                                    padding: 20px; border-radius: 10px; color: #e0e0e0; margin-bottom: 20px;">
                            <h4 style="color: #00d4aa; margin-top: 0;">Expert Plant Pathologist Advice</h4>
                            <p style="line-height: 1.6;">{treatment_advice.replace(chr(10), '<br>')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(200, 10, txt="AI Treatment Recommendation:", ln=True)

                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 7, txt=treatment_advice)

                    pdf.ln(5)

                    # Now generate your pdf_bytes
                    pdf_bytes = pdf.output(dest="S").encode("latin-1")
                    
                    # 5. Download Button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Plant_Report_{ts}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    elif page == "⚙️ Settings":
        st.header("⚙️ Account Settings")
        c.execute("SELECT first_name, last_name, password, profile_pic FROM users WHERE email=?", (st.session_state.user['email'],))
        u = c.fetchone()
        
        with st.form("settings_form"):
            new_fname = st.text_input("First Name", value=u[0])
            new_lname = st.text_input("Last Name", value=u[1])
            new_pwd = st.text_input("New Password (leave blank to keep current)", type="password")
            new_pic = st.file_uploader("Update Profile Picture", type=['png', 'jpg'])
            
            if st.form_submit_button("Save Changes"):
                pic_path = u[3]
                if new_pic:
                    pic_path = f"{UPLOAD_FOLDER}/profiles/{st.session_state.user['email']}_{new_pic.name}"
                    with open(pic_path, "wb") as f: f.write(new_pic.getbuffer())
                
                pwd = new_pwd if new_pwd else u[2]
                c.execute("UPDATE users SET first_name=?, last_name=?, password=?, profile_pic=? WHERE email=?", 
                          (new_fname, new_lname, pwd, pic_path, st.session_state.user['email']))
                conn.commit()
                
                # --- THIS IS THE FIX ---
                # Manually update the session state so the UI reflects changes instantly
                st.session_state.user['name'] = f"{new_fname} {new_lname}"
                
                st.success("Profile updated successfully!")
                st.rerun()

#------------------HISTORY RECORDS--------------#
    elif page == "📜 Prediction History":
        st.title("📜 Prediction History")
        
        # 1. Summary Chart
        df_data = c.execute("SELECT disease, count(*) FROM predictions WHERE user_email=? GROUP BY disease", 
                            (st.session_state.user["email"],)).fetchall()
        if df_data:
            import pandas as pd
            st.write("### Disease Detection Trends")
            
            st.bar_chart(pd.DataFrame(df_data, columns=['Disease', 'Count']).set_index('Disease'))
        
        # 2. Detailed History List
        c.execute("SELECT rowid, leaf_image, disease, confidence, created_at FROM predictions WHERE user_email=? ORDER BY created_at DESC", 
                  (st.session_state.user["email"],))
        history = c.fetchall()
        
        if not history:
            st.warning("No previous detections found.")
        else:
            for r in history:
                row_id, img_path, disease, conf_val, created_at = r
                
                # Format the time to HH:MM:SS
                # Robust Time Parsing for 24-hour format
                try:
                    # Ensure created_at is a string and parse it using the standard database format
                    # %H ensures 24-hour clock (00-23)
                    dt = datetime.strptime(str(created_at), "%Y-%m-%d %H:%M:%S")
                    formatted_time = dt.strftime("%H:%M:%S") 
                    display_date = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    # Fallback if the database format is unexpected
                    formatted_time = str(created_at)
                    display_date = ""

                # Safely parse confidence
                try:
                    safe_conf = float(conf_val) if not isinstance(conf_val, bytes) else float(conf_val.decode('utf-8', 'ignore'))
                except:
                    safe_conf = 0.0

                with st.expander(f"📅 {display_date} | ⏰ {formatted_time} | 🎯 {disease}"):
                    col1, col2 = st.columns([1, 2])
                    
                    # Display Image
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            col1.image(f.read(), use_container_width=True)
                    else:
                        col1.error("Image missing")
                        
                    # Display Details
                    col2.markdown(f"### {disease}")
                    col2.metric("Confidence Score", f"{safe_conf:.2f}%")
                    col2.write(f"**Prediction Time:** {formatted_time}")
                    
                    # Delete Record
                    if col2.button("🗑️ Delete Record", key=f"del_{row_id}"):
                        c.execute("DELETE FROM predictions WHERE rowid=?", (row_id,))
                        conn.commit()
                        st.rerun()
    elif page == "👤 Account Profile":
        st.title("👤 Account Profile")
        c.execute("SELECT first_name, last_name, email, profile_pic, created_at FROM users WHERE email=?", (st.session_state.user["email"],))
        u = c.fetchone()
        
        # Process the date (u[4] is the 5th column: created_at)
        # This assumes the format 'YYYY-MM-DD HH:MM:SS'
        raw_date = str(u[4])
        display_date = raw_date.split(' ')[0] # This takes everything before the space

        with st.container(border=True):
            col_a, col_b = st.columns([1, 2])
            
            # Profile Image
            if u[3] and os.path.exists(u[3]):
                with open(u[3], "rb") as f: 
                    col_a.image(f.read(), use_container_width=True)
            else:
                col_a.write("No image available")
                
            # Profile Details
            col_b.subheader(f"{u[0]} {u[1]}")
            col_b.write(f"📧 **Email:** {u[2]}")
            col_b.write(f"📅 **Joined:** {display_date}")

    elif page == "🚪 Logout":
        st.session_state.clear()
        st.rerun()