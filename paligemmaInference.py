from autodistill_paligemma import PaliGemma
from autodistill.detection import CaptionOntology

base_model = PaliGemma(ontology=CaptionOntology({"A person in black shirt":"Black Shirt"}))

base_model.label("D:\\Datasets\\COCO Test60 - Copy\\",batch_size=4) 