import streamlit as st
import tensorflow as tf
import numpy as np

# 🌱 Page Config
st.set_page_config(page_title="Mango Disease Recognition", page_icon="🌿", layout="wide")

# 🧠 Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# 🎯 Model Prediction Function (Now Returns Class Index & Confidence)
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(input_arr)[0]  # Get probability array
    max_prob = np.max(predictions)  # Highest confidence score
    pred_class_index = np.argmax(predictions)  # Class index of highest score

    return pred_class_index, max_prob, predictions  # Return all probabilities

# 🌍 Sidebar Navigation
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("📌 Select Page", ["🏠 Home", "🔬 Disease Recognition"])

# 📌 Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌱 MANGO DISEASE RECOGNITION SYSTEM 🔍</h1>", unsafe_allow_html=True)
    st.image("https://source.unsplash.com/800x400/?plant,leaves", use_container_width=True)

    st.markdown("""
    ### 🌟 Welcome to the Mango Disease Recognition System!
    Our mission is to help **farmers & researchers** detect plant diseases with **AI-powered image recognition**.  

    🔹 **How It Works:**  
    1️⃣ Upload an image of a plant leaf  
    2️⃣ The AI model will analyze the image  
    3️⃣ Get instant results on the plant’s health  

    🔹 **Why Choose Us?**  
    ✅ High accuracy with deep learning  
    ✅ User-friendly & intuitive interface  
    ✅ Fast results for quick decision-making  
    """)
    
    st.info("➡️ Go to the **Disease Recognition** page to upload an image!")

# 🔬 Disease Recognition Page
elif app_mode == "🔬 Disease Recognition":
    st.markdown("<h1 style='text-align: center;'>🔍 Disease Recognition</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])  # 🟢 Use columns for better layout

    with col1:
        test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "png", "jpeg"])

        if test_image is not None:
            st.session_state["uploaded_image"] = test_image

        # If image is uploaded, display it
        if "uploaded_image" in st.session_state:
            st.image(st.session_state["uploaded_image"], caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🚀 Predict") and "uploaded_image" in st.session_state:
            st.snow()
            st.markdown("### 🤖 AI Prediction:")

            result_index, confidence, all_probs = model_prediction(st.session_state["uploaded_image"])

            # Disease Labels (Make Sure This Matches Your Training Order!)
            class_name = ['Anthracnose(fruit)', 'Healthy(fruit)', 'Anthracnose(leaf)', 'Powdery Mildew', 'Healthy']

            # Confidence Threshold Check
            if confidence < 0.5:  # Less than 50% confidence → Not a mango leaf
                st.error("🚫 This image is **not recognized as a mango plant**. Please upload a valid mango plant image! 🌿")
            else:
                disease = class_name[result_index]
                if "Healthy" in disease:
                    st.success(f"🌿 Your plant is **Healthy**! Good job! 🍃✅ (Confidence: {confidence:.2%})")
                else:
                    st.warning(f"⚠️ The plant shows signs of **{disease}**. Please take action! 🚨 (Confidence: {confidence:.2%})")
                    if disease == "Anthracnose(leaf)" or "Anthracnose(fruit)":
                         st.info(  """Treatment & Management:

                                    •Spray copper-based fungicides (e.g., Copper oxychloride, Bordeaux mixture, Antracol) at the flowering stage.

                                    •Use systemic fungicides like Carbendazim, Thiophanate-methyl, or Azoxystrobin during fruit development.

                                    •Apply Chlorothalonil or Mancozeb at early flowering and fruit set stages.""")

            # Show Class Probabilities
            st.subheader("📊 Confidence Scores:")
            for i, prob in enumerate(all_probs):
                st.write(f"🔹 {class_name[i]}: **{prob:.4f}**")
          
