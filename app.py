import streamlit as st
from PIL import Image
import model

st.set_page_config(page_title="Toxic Comment Classifier")

def main():
    css = """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        color: #333333;
    }
    .container {
        max-width: 800px;
        padding: 20px;
        margin: auto;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .title-box {
        padding: 15px;
        background-color: #0072b5;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }
    .title-text {
        color: black;
        font-size: 30px;
        font-weight: bold;
        text-transform: uppercase;
        text-align: center;
    }
    .input-section {
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .output-section {
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    html_temp = """
    <div class="title-box">
        <h2 class="title-text">Toxic Comment Classifier</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    option = st.sidebar.selectbox("Select Option", ["Toxic Comment Classifier", "Image Uploader"])

    if option == "Toxic Comment Classifier":
        toxic_comment_classifier()
    elif option == "Image Uploader":
        image_uploader()

def toxic_comment_classifier():
    st.markdown("<h3 class='input-section'>Enter The Comment</h3>", unsafe_allow_html=True)
    comment = st.text_area("Comment", "", height=150)

    button_clicked = st.button("Predict Toxicity")

    if button_clicked and comment.strip():
        contractions_removed = model.expand_contraction(comment)
        spell_corrected = model.spell_check(contractions_removed)
        processed_script = model.clean_text(spell_corrected)
        prediction = model.toxicity_prediction(processed_script)

        st.markdown("<div class='output-section'>", unsafe_allow_html=True)

        if prediction * 100 < 30:
            st.markdown(f"### Comment: {processed_script}")
            st.success("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))
        elif 30 <= prediction * 100 < 70:
            st.markdown(f"### Comment: {processed_script}")
            st.warning("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))
        else:
            st.markdown(f"### Comment: {processed_script}")
            st.error("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))

        st.markdown("</div>", unsafe_allow_html=True)
    
    elif button_clicked:
        st.error('#### Enter the text')



def image_uploader():
    st.subheader("Image Uploader and Viewer")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    count = 0
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        path = uploaded_file.name
        count +=1
    
    button_clicked = st.button("Predict Toxicity")

    if button_clicked and count!=0:

        image_text = model.extract_text_from_image(path)
        contractions_removed = model.expand_contraction(image_text)
        spell_corrected = model.spell_check(contractions_removed)
        processed_script = model.clean_text(spell_corrected)
        prediction = model.toxicity_prediction(processed_script)

        st.markdown("<div class='output-section'>", unsafe_allow_html=True)

        if prediction * 100 < 30:
            st.markdown(f"### Comment: {processed_script}")
            st.success("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))
        elif 30 <= prediction * 100 < 70:
            st.markdown(f"### Comment: {processed_script}")
            st.warning("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))
        else:
            st.markdown(f"### Comment: {processed_script}")
            st.error("#### Comment Toxicity: {}%".format(round(prediction[0] * 100),2))

        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.error('#### Select the image')

if __name__ == "__main__":
    main()
