from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

base_model = GroundingDINO(ontology=CaptionOntology({"A person in black shirt":"Black Shirt"}))

base_model.label("D:\Datasets\COCO Test60 - Copy", extension =".jpg",batch_size=10)

# from autodistill_grounding_dino import GroundingDINO
# from autodistill.detection import CaptionOntology

# # Define an ontology to map class names to our GroundingDINO prompt
# ontology = CaptionOntology({"A person in black shirt":"Black Shirt"})

# # Load the model with the defined ontology
# base_model = GroundingDINO(ontology=ontology)

# # Check the weights
# def check_model_weights(model):
#     # This function assumes the model has an accessible attribute for weights
#     # Modify it according to your model's actual implementation
#     for name, param in model.named_parameters():
#         print(f"Layer: {name}, Weights: {param}")

# check_model_weights(base_model.model)  # Assuming base_model.model is the actual model with weights
