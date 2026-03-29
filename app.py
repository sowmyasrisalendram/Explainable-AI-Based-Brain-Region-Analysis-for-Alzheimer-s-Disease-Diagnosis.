import pytz
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer 
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

# -------- Page Config -------- #
st.set_page_config(page_title="Alzheimer’s Diagnosis", layout="centered")

# -------- Load Model -------- #
model = tf.keras.models.load_model("alzheimers_model.keras")

class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# -------- Custom CSS -------- #
st.markdown("""
<style>

/* Center container */
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}

/* Background */
.stApp {
    background-color: #f4f8fb;
}

/* Title */
h1 {
    text-align:center;
    color:#0b5394;
    font-size: 30px;
    white-space: nowrap;
}

/* Upload box */
[data-testid="stFileUploader"]{
    border:2px dashed #0b5394;
    padding:10px;
    border-radius:10px;
    max-width:700px;
    margin:auto;
}

</style>
""", unsafe_allow_html=True)



# -------- Title -------- #
st.markdown("""
<div style="text-align:center">

<h1 style="font-size:42px; margin-bottom:5px;">
 Explainable AI – Alzheimer’s Diagnosis System
</h1>

<p style="font-size:18px; color:#555;">
AI-powered MRI classification with visual brain region explanation using Grad-CAM
</p>

<hr style="margin-top:25px; margin-bottom:25px;">

</div>
            
""", unsafe_allow_html=True)


# -------- Grad-CAM Function -------- #
def make_gradcam_heatmap(img_array, model):

    last_conv_layer = None

    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# -------- Patient Information -------- #
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")

with col2:
    patient_age = st.number_input("Patient Age", 1, 120)

st.markdown("---")

# -------- Upload MRI -------- #
uploaded_file = st.file_uploader(
"Upload Brain MRI Image",
type=["jpg","png","jpeg"]
)


# -------- Prediction -------- #
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128,128))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # -------- Grad-CAM -------- #
    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (128,128))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    # -------- Display -------- #
    # -------- MRI + GradCAM Section -------- #

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2,gap="large")

    with col1:
        st.markdown(
            "<h3 style='font-size:20px; margin-left: 100px;'>Uploaded MRI</h3>",
            unsafe_allow_html=True
        )
        st.image(img, width=350)
            


    with col2:
        st.markdown("<h3 style='font-size:20px; margin-left: 50px;'>Alzheimer’s Region Heatmap</h3>", 
            unsafe_allow_html=True)
        st.image(superimposed_img, width=350)


    # -------- Divider before Diagnosis -------- #
    st.markdown("---")

    st.markdown(
        "<h3 style='text-align:center;'>AI Diagnosis</h3>",
        unsafe_allow_html=True
    )

    st.success(f"Predicted Class : {predicted_class}")
    st.write("### Model Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}% Confidence")



    st.markdown("---")

    
    # -------- Heatmap Explanation -------- #
    st.subheader(" Heatmap Explanation")

    st.write("""
    🔴 **Red / Yellow regions** → Brain areas that the AI model considered **highly important** for making the diagnosis.

    🟠 **Orange regions** → **Moderately important regions** that contributed to the model’s decision.

    🔵 **Blue regions** → Brain regions with **minimal influence** on the prediction.

    The highlighted areas represent brain regions where the model detected significant structural patterns in the MRI scan.  
    These patterns may include **brain tissue shrinkage, cortical thinning, or other structural changes** commonly associated with Alzheimer’s disease progression.

    """)

    st.markdown("---")


    # -------- Interpretation -------- #
    st.subheader("Interpretation")

    if predicted_class == "Non Demented":
        recommendation = "No Alzheimer’s patterns detected. Regular monitoring recommended."

    elif predicted_class == "Very Mild Demented":
        recommendation = "Early cognitive changes detected. Clinical evaluation recommended."

    elif predicted_class == "Mild Demented":
        recommendation = "Mild Alzheimer’s stage detected. Neurologist consultation advised."

    else:
        recommendation = "Moderate Alzheimer’s stage detected. Immediate specialist consultation required."

    st.warning(recommendation)

    st.markdown("---")

    
    # -------- PDF Report -------- #
    st.subheader("Download Medical Report")

    if st.button("Generate PDF Report"):

        from datetime import datetime
        from reportlab.lib.styles import ParagraphStyle
        from datetime import datetime

        pdf_path = "Alzheimer_Report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)

        styles = getSampleStyleSheet()

        header_style = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        alignment=2,   # Right align
        fontSize=10
    )

        positive_style = ParagraphStyle(
        'Positive',
        parent=styles['Normal'],
        #backColor='#ffe6e6',   # light red
        textColor='red',
        fontSize=12,
    )

        negative_style = ParagraphStyle(
        'Negative',
        parent=styles['Normal'],
        #backColor='#e6ffe6',   # light green
        textColor='green',
        fontSize=12,
    )
        

        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        report_date = now.strftime("%d %B %Y")   # Example: 06 March 2026
        report_time = now.strftime("%I:%M %p IST")   # Example: 10:42:15

        elements = []

        # Date and Time at top-right
        elements.append(Paragraph(f"Date : {report_date}", header_style))
        elements.append(Paragraph(f"Time : {report_time}", header_style))


        elements.append(Spacer(1,20))


        # Title
        elements.append(Paragraph("<b>Alzheimer’s Disease AI Diagnosis Report</b>", styles['Title']))
        elements.append(Spacer(1, 20))

        # Patient Information
        elements.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
        elements.append(Spacer(1,10))
        elements.append(Paragraph(f"Patient Name : {patient_name}", styles['Normal']))
        elements.append(Paragraph(f"Age : {patient_age}", styles['Normal']))
        elements.append(Spacer(1,20))

        # MRI Scan Info
        elements.append(Paragraph("<b>MRI Scan Information</b>", styles['Heading2']))
        elements.append(Spacer(1,10))
        elements.append(Paragraph("Scan Type : Brain MRI", styles['Normal']))
        elements.append(Paragraph("AI Model : Convolutional Neural Network (CNN)", styles['Normal']))
        elements.append(Spacer(1,20))

        # Diagnosis
        elements.append(Paragraph("<b>AI Diagnosis Result</b>", styles['Heading2']))
        elements.append(Spacer(1,10))
        elements.append(Paragraph(f"Diagnosis : {predicted_class}", styles['Normal']))
        elements.append(Paragraph(f"Confidence : {confidence:.2f} %", styles['Normal']))
        elements.append(Spacer(1,15))

        #Diagnosis Status
        elements.append(Paragraph("<b>Diagnosis Status</b>", styles['Heading2']))
        elements.append(Spacer(1,10))

        if predicted_class == "Non Demented":
            status = "Negative for Alzheimer’s Disease"
            elements.append(Paragraph(f"<b>{status}</b>", negative_style))

        else:
            status = "Positive for Alzheimer’s Disease"
            elements.append(Paragraph(f"<b>{status}</b>", positive_style))

        elements.append(Spacer(1,20))

        # Interpretation
        elements.append(Paragraph("<b>Interpretation</b>", styles['Heading2']))
        elements.append(Paragraph(recommendation, styles['Normal']))
        elements.append(Spacer(1,20))


        # -------- Clinical Diagnosis -------- #
        elements.append(Paragraph("<b>Clinical Diagnosis</b>", styles['Heading2']))
        elements.append(Spacer(1,10))

        if predicted_class == "Non Demented":

            diagnosis_text = """
            The MRI scan analysis shows no significant structural abnormalities 
            associated with Alzheimer’s disease. Brain regions appear normal with 
            no visible patterns of cognitive degeneration. The patient is currently 
            classified as neurologically healthy based on the AI evaluation.
            """

        elif predicted_class == "Very Mild Demented":

            diagnosis_text = """
            The MRI scan indicates very early structural brain changes that may be 
            associated with the initial stages of cognitive decline. These findings 
            may represent early indicators of Alzheimer’s disease and require 
            periodic monitoring and clinical evaluation.
            """

        elif predicted_class == "Mild Demented":

            diagnosis_text = """
            The MRI scan reveals patterns of brain tissue changes consistent with 
            mild Alzheimer’s disease. Certain brain regions associated with memory 
            and cognitive processing show signs of degeneration, suggesting early 
            disease progression.
            """

        else:   # Moderate Demented

            diagnosis_text = """
            The MRI scan demonstrates significant structural brain changes 
            associated with moderate Alzheimer’s disease. The model detected 
            pronounced degeneration patterns in cognitive-related brain regions, 
            indicating an advanced stage of neurodegenerative progression.
            """

        elements.append(Paragraph(diagnosis_text, styles['Normal']))
        elements.append(Spacer(1,20))


        # -------- Prescription / Medical Advice -------- #
        elements.append(Paragraph("<b>Prescription / Recommendation</b>", styles['Heading2']))
        elements.append(Spacer(1,10))

        if predicted_class == "Non Demented":
            prescription = "No medication required. Maintain healthy lifestyle and regular neurological check-ups."

        elif predicted_class == "Very Mild Demented":
            prescription = "Regular cognitive monitoring and medical consultation recommended."

        elif predicted_class == "Mild Demented":
            prescription = "Neurologist consultation and cognitive therapy evaluation recommended."

        else:
            prescription = "Immediate specialist consultation and comprehensive neurological assessment advised."

        elements.append(Paragraph(prescription, styles['Normal']))
        elements.append(Spacer(1,20))

        doc.build(elements)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                "Alzheimer_Report.pdf"
            )