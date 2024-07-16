from autodistill_gemini import Gemini
from autodistill.detection import CaptionOntology

base_model = Gemini(ontology=CaptionOntology({'A person in black shirt':'Black Shirt'}))

result = base_model.predict("D:\\Datasets\\COCO Test60_labeled\\train\\images\\000000001732_jpg.rf.8c7233adcaafa3f8801a80435c7cefca.jpg")

print(result)