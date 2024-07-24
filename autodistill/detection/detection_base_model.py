# import enum
# import glob
# import os
# from abc import abstractmethod
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List
# import cv2
# import numpy as np
# import time 
# from tqdm import tqdm
# from autodistill.core import BaseModel
# from autodistill.helpers import load_image, split_data
# import supervision as sv
# from .detection_ontology import DetectionOntology


# class NmsSetting(str, enum.Enum):
#     NONE = "no_nms"
#     CLASS_SPECIFIC = "class_specific"
#     CLASS_AGNOSTIC = "class_agnostic"


# @dataclass
# class DetectionBaseModel(BaseModel):
#     ontology: DetectionOntology

#     @abstractmethod
#     def predict(self, input: str | np.ndarray) -> sv.Detections:
#         pass

#     def sahi_predict(self, input: str | np.ndarray) -> sv.Detections:
#         slicer = sv.InferenceSlicer(callback=self.predict)
#         return slicer(load_image(input, return_format="cv2"))

#     def _record_confidence_in_files(
#         self,
#         annotations_directory_path: str,
#         images: Dict[str, np.ndarray],
#         annotations: Dict[str, sv.Detections],
#     ) -> None:
#         Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
#         for image_name, _ in images.items():
#             detections = annotations[image_name]
#             yolo_annotations_name, _ = os.path.splitext(image_name)
#             confidence_path = os.path.join(
#                 annotations_directory_path,
#                 "confidence-" + yolo_annotations_name + ".txt",
#             ) 
#             confidence_list = [str(x) for x in detections.confidence.tolist()]
#             save_text_file(lines=confidence_list, file_path=confidence_path)
#             print("Saved confidence file: " + confidence_path)

#     def label(
#         self,
#         input_folder: str,
#         extension: str = ".jpg",
#         output_folder: str = None,
#         human_in_the_loop: bool = False,
#         roboflow_project: str = None,
#         roboflow_tags: List[str] = ["autodistill"],
#         sahi: bool = False,
#         record_confidence: bool = False,
#         nms_settings: NmsSetting = NmsSetting.NONE,
#         # batch_size: int = 2 # Added batch_size parameter
#     ) -> sv.DetectionDataset:
#         """
#         Label a dataset with the model.
#         """
#         if output_folder is None:
#             output_folder = input_folder + "_labeled"

#         os.makedirs(output_folder, exist_ok=True)

#         images_map = {}
#         detections_map = {}

#         if sahi:
#             slicer = sv.InferenceSlicer(callback=self.predict)

#         files = glob.glob(input_folder + "/*" + extension)
        
        
#         # UNCOMMENT FROM HERE TO LINE NO 103 to run on whole folder.
#         progress_bar = tqdm(files, desc="Labeling images")
        
#         start_time = time.time()
#         # iterate through images in input_folder
#         for f_path in progress_bar:
#             progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
#             image = cv2.imread(f_path)

#             f_path_short = os.path.basename(f_path)
#             images_map[f_path_short] = image.copy()

#             if sahi:
#                 detections = slicer(image)
        
#             else:
#                 detections = self.predict(image)

#             if nms_settings == NmsSetting.CLASS_SPECIFIC:
#                 detections = detections.with_nms()
#             if nms_settings == NmsSetting.CLASS_AGNOSTIC:
#                 detections = detections.with_nms(class_agnostic=True)

#             detections_map[f_path_short] = detections

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         total_time = elapsed_time//60
#         print(f"Time take to label the dataset {total_time} minitues") 
        
        
#         # BATCH INFERENCE (UNCOMMENT FROM HERE TO LINE NO 152 TO USE BATCH INFERENCE)
        
#         # def batch_loader(file_list, batch_size):
#         #     for i in range(0, len(file_list), batch_size):
#         #         yield file_list[i:i + batch_size]

#         # total_batches = (len(files) + batch_size - 1) // batch_size
#         # main_progress_bar = tqdm(total=total_batches, desc="Processing batches")

#         # start_time = time.time()  
#         # for batch_index,batch_files in enumerate(batch_loader(files, batch_size),start=1):
#         #     batch_images = []
#         #     for file_path in batch_files:
#         #         image = cv2.imread(file_path)
#         #         if image is not None:
#         #             batch_images.append(image)
            
#         #     batch_desc = f"Inferencing Batch-{batch_index}"
#         #     batch_progress_bar = tqdm(total=len(batch_images), desc=batch_desc, leave=False)

#         #     if sahi:
#         #         detections_batch = []
#         #         for image in batch_images:
#         #             detections = slicer(image)
#         #             detections_batch.append(detections)
#         #             batch_progress_bar.update(1)
#         #     else:
#         #         detections_batch = []
#         #         for image in batch_images:
#         #             detections = self.predict(image)
#         #             detections_batch.append(detections)
#         #             batch_progress_bar.update(1)

#         #     batch_progress_bar.close()

#         #     for img_path, image, detections in zip(batch_files, batch_images, detections_batch):
#         #         img_path_short = os.path.basename(img_path)
#         # #         images_map[img_path_short] = image

#         #         if nms_settings == NmsSetting.CLASS_SPECIFIC:
#         #             detections = detections.with_nms()
#         #         if nms_settings == NmsSetting.CLASS_AGNOSTIC:
#         #             detections = detections.with_nms(class_agnostic=True)

#         #         detections_map[img_path_short] = detections

#         #     main_progress_bar.update(1)

#         # main_progress_bar.close()
        
        
#         # end_time = time.time()
#         # elapsed_time = end_time - start_time
#         # print(f"Time take to label the dataset {elapsed_time:.5f} seconds") 
        
#         dataset = sv.DetectionDataset(
#             self.ontology.classes(), images_map, detections_map
#         )

#         dataset.as_yolo(
#             output_folder + "/images",
#             output_folder + "/annotations",
#             min_image_area_percentage=0.01,
#             data_yaml_path=output_folder + "/data.yaml",
#         )

#         if record_confidence is True:
#             self._record_confidence_in_files(
#                 output_folder + "/annotations", images_map, detections_map
#             )
#         split_data(output_folder, record_confidence=record_confidence)

#         if human_in_the_loop:
#             roboflow.login()

#             rf = roboflow.Roboflow()

#             workspace = rf.workspace()

#             workspace.upload_dataset(output_folder, project_name=roboflow_project)

#         print("Labeled dataset created - ready for distillation.")
#         return dataset


import enum
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import roboflow
import supervision as sv
from PIL import Image
from supervision.utils.file import save_text_file
from tqdm import tqdm


from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology


class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        pass

    def sahi_predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)
        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        images: Dict[str, np.ndarray],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name, _ in images.items():
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)

    def label(
        self,
        input_folder: str, 
        extension: str = ".jpg",
        output_folder: str = None,
        human_in_the_loop: bool = False,
        roboflow_project: str = None,
        roboflow_tags: str = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
        batch_size: int = 10,
        resize_to: tuple = (224, 224)
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        images_map = {}
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")

        # Prepare a batcher class
        class ImageBatcher:
            def __init__(self, batch_size):
                self.batch_size = batch_size
                self.current_batch = []
                self.file_paths = []

            def add_image(self, image, f_path):
                self.current_batch.append(image)
                self.file_paths.append(f_path)
                if len(self.current_batch) == self.batch_size:
                    batch = (self.current_batch, self.file_paths)
                    self.current_batch = []
                    self.file_paths = []
                    return batch
                return None

            def get_remaining(self):
                if self.current_batch:
                    batch = (self.current_batch, self.file_paths)
                    self.current_batch = []
                    self.file_paths = []
                    return batch
                return None

        batcher = ImageBatcher(batch_size)

        # Iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)
            image_resized = cv2.resize(image, resize_to)
            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image_resized.copy()

            batch = batcher.add_image(image_resized, f_path_short)
            if batch:
                batch_images, batch_paths = batch
                batch_images = np.stack(batch_images)
                print(type(batch))
                print(batch)

                if sahi:
                    detections_batch = slicer(batch_images)
                else:
                    detections_batch = self.predict(batch_images)

                for img_path, detections in zip(batch_paths, detections_batch):
                    if nms_settings == NmsSetting.CLASS_SPECIFIC:
                        detections = detections.with_nms()
                    if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                        detections = detections.with_nms(class_agnostic=True)
                    detections_map[img_path] = detections

        # Process remaining images
        remaining_batch = batcher.get_remaining()
        if remaining_batch:
            batch_images, batch_paths = remaining_batch
            batch_images = np.stack(batch_images)

            if sahi:
                detections_batch = slicer(batch_images)
            else:
                detections_batch = self.predict(batch_images)

            for img_path, detections in zip(batch_paths, detections_batch):
                if nms_settings == NmsSetting.CLASS_SPECIFIC:
                    detections = detections.with_nms()
                if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                    detections = detections.with_nms(class_agnostic=True)
                detections_map[img_path] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), images_map, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )
        
        if record_confidence is True:
            self._record_confidence_in_files(
                output_folder + "/annotations", images_map, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset


# import enum
# import glob
# import os
# from abc import abstractmethod
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Tuple

# import cv2
# import numpy as np
# import roboflow
# import supervision as sv
# from PIL import Image
# from supervision.utils.file import save_text_file
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# import torch

# from autodistill.core import BaseModel
# from autodistill.helpers import load_image, split_data

# from .detection_ontology import DetectionOntology


# class NmsSetting(str, enum.Enum):
#     NONE = "no_nms"
#     CLASS_SPECIFIC = "class_specific"
#     CLASS_AGNOSTIC = "class_agnostic"


# @dataclass
# class DetectionBaseModel(BaseModel):
#     ontology: DetectionOntology

#     @abstractmethod
#     def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
#         pass

#     def sahi_predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
#         slicer = sv.InferenceSlicer(callback=self.predict)
#         return slicer(load_image(input, return_format="cv2"))

#     def _record_confidence_in_files(
#         self,
#         annotations_directory_path: str,
#         images: Dict[str, np.ndarray],
#         annotations: Dict[str, sv.Detections],
#     ) -> None:
#         Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
#         for image_name, _ in images.items():
#             detections = annotations[image_name]
#             yolo_annotations_name, _ = os.path.splitext(image_name)
#             confidence_path = os.path.join(
#                 annotations_directory_path,
#                 "confidence-" + yolo_annotations_name + ".txt",
#             )
#             confidence_list = [str(x) for x in detections.confidence.tolist()]
#             save_text_file(lines=confidence_list, file_path=confidence_path)
#             print("Saved confidence file: " + confidence_path)

#     def label(
#         self,
#         input_folder: str,
#         extension: str = ".jpg",
#         output_folder: str = None,
#         human_in_the_loop: bool = False,
#         roboflow_project: str = None,
#         roboflow_tags: str = ["autodistill"],
#         sahi: bool = False,
#         record_confidence: bool = False,
#         nms_settings: NmsSetting = NmsSetting.NONE,
#         batch_size: int = 32,
#         resize_to: Tuple[int, int] = (224, 224)
#     ) -> sv.DetectionDataset:
#         """
#         Label a dataset with the model.
#         """
#         if output_folder is None:
#             output_folder = input_folder + "_labeled"

#         os.makedirs(output_folder, exist_ok=True)

#         images_map = {}
#         detections_map = {}

#         if sahi:
#             slicer = sv.InferenceSlicer(callback=self.predict)

#         files = glob.glob(input_folder + "/*" + extension)
#         progress_bar = tqdm(files, desc="Labeling images")

#         def process_batch(batch: Tuple[List[np.ndarray], List[str]]) -> Dict[str, sv.Detections]:
#             batch_images, batch_paths = batch
#             batch_images = np.stack(batch_images)
#             batch_images_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2)  # Convert to tensor and permute to (batch_size, channels, height, width)

#             if sahi:
#                 detections_batch = slicer(batch_images_tensor)
#             else:
#                 detections_batch = self.predict(batch_images_tensor)

#             results = {}
#             for img_path, detections in zip(batch_paths, detections_batch):
#                 if nms_settings == NmsSetting.CLASS_SPECIFIC:
#                     detections = detections.with_nms()
#                 if nms_settings == NmsSetting.CLASS_AGNOSTIC:
#                     detections = detections.with_nms(class_agnostic=True)
#                 results[img_path] = detections
#             return results

#         class ImageBatcher:
#             def __init__(self, batch_size):
#                 self.batch_size = batch_size
#                 self.current_batch = []
#                 self.file_paths = []

#             def add_image(self, image, f_path):
#                 self.current_batch.append(image)
#                 self.file_paths.append(f_path)
#                 if len(self.current_batch) == self.batch_size:
#                     batch = (self.current_batch, self.file_paths)
#                     self.current_batch = []
#                     self.file_paths = []
#                     return batch
#                 return None

#             def get_remaining(self):
#                 if self.current_batch:
#                     batch = (self.current_batch, self.file_paths)
#                     self.current_batch = []
#                     self.file_paths = []
#                     return batch
#                 return None

#         batcher = ImageBatcher(batch_size)
#         batches = []

#         for f_path in progress_bar:
#             progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
#             image = cv2.imread(f_path)
#             image_resized = cv2.resize(image, resize_to)
#             f_path_short = os.path.basename(f_path)
#             images_map[f_path_short] = image_resized.copy()

#             batch = batcher.add_image(image_resized, f_path_short)
#             if batch:
#                 batches.append(batch)

#         remaining_batch = batcher.get_remaining()
#         if remaining_batch:
#             batches.append(remaining_batch)

#         # Use multiprocessing to process batches
#         with Pool(cpu_count()) as pool:
#             results = pool.map(process_batch, batches)

#         for result in results:
#             detections_map.update(result)

#         dataset = sv.DetectionDataset(
#             self.ontology.classes(), images_map, detections_map
#         )

#         dataset.as_yolo(
#             output_folder + "/images",
#             output_folder + "/annotations",
#             min_image_area_percentage=0.01,
#             data_yaml_path=output_folder + "/data.yaml",
#         )

#         if record_confidence is True:
#             self._record_confidence_in_files(
#                 output_folder + "/annotations", images_map, detections_map
#             )
#         split_data(output_folder, record_confidence=record_confidence)

#         if human_in_the_loop:
#             roboflow.login()

#             rf = roboflow.Roboflow()

#             workspace = rf.workspace()

#             workspace.upload_dataset(output_folder, project_name=roboflow_project)

#         print("Labeled dataset created - ready for distillation.")
#         return dataset
