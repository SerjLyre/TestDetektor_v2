#|export
import fastbook
import gradio as gr
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

learn = load_learner('.\\GD_MDL_RESNET34_Finetune_5.pkl')

categories = ('Gali', 'Nicht_Gali')
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['71_examples\\NichtGali.jpg', '71_examples\\NichtGali2.jpg', '71_examples\\NichtGali3.jpg']
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title="Gali Detektor")
intf.launch(inline=False)
