from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st


@st.cache_resource
class AppModel: 

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained("./output/model/checkpoint-2500/")

    def generate_plot(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        output_string = self.tokenizer.batch_decode(outputs)

        return output_string

model = AppModel()

prompt = st.text_area("Enter the beginning of your plot...")
clicked = st.button("Generate my plot!")

if clicked:
    generated_plot = model.generate_plot(prompt)[0]

    chat_message = st.chat_message("assistant")
    chat_message.markdown(generated_plot)
