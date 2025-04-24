import requests
import pandas as pd
from io import BytesIO
from PIL import Image


def get_sample_data(document_id: int):
    get_samples_url = (
        f"http://localhost:8000/document_app/api/documents/{document_id}/get_samples/"
    )

    with requests.Session() as session:
        res_samples = session.get(get_samples_url)

    if res_samples.status_code != 200:
        raise ValueError(f"cannot retrieve samples for document: {document_id}")
    else:
        samples = res_samples.json()

    all_boxes = []

    for sample in samples:
        sample_id = sample["id"]
        get_sample_boxes_url = f"http://localhost:8000/document_app/api/sample_documents/{sample_id}/get_boxes/"

        with requests.Session() as session:
            res_boxes = session.get(get_sample_boxes_url)

        if res_boxes.status_code != 200:
            raise ValueError(f"cannot retrieve boxes for sample: {sample_id}")
        else:
            boxes = res_boxes.json()

        all_boxes += boxes

    box_labels = pd.DataFrame(all_boxes)

    images = []
    for sample in samples:
        with requests.Session() as session:
            res_image = session.get("http://localhost:8000/" + sample["image"])

        if res_image.status_code == 200:
            sample_image = Image.open(BytesIO(res_image.content))
        else:
            raise ValueError(
                f"could not retrieve the image error: {res_image.status_code}"
            )

        images.append({"sample_id": sample["id"], "image": sample_image})

    return images, box_labels
