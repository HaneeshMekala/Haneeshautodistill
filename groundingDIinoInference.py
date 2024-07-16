from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

base_model = GroundingDINO(ontology=CaptionOntology({"A person in black shirt":"Black Shirt"}))

base_model.label("D:\\Datasets\\COCO Test60withoutBatch\\", extension =".jpg", batch_size=10)

