import requests
import logging


def get_image_and_boxes(document_id: str):
    with requests.Session() as session:
        res_document = session.get(f"http://localhost:8000/document_generator/api/documents/{document_id}")
        res_boxes = session.get(f"http://localhost:8000/document_generator/api/documents/{document_id}/get_boxes/")
    
        if res_document.status_code == 200 and res_boxes.status_code == 200:
            document_image_url = res_document.json()["image"]
            boxes = res_boxes.json()
            res_image = session.get(document_image_url)
    
            if res_image.status_code == 200:
                document_image_buffer = res_image.content
            else:
                raise ValueError(f"could not retrieve the image error: {res_image.status_code}")
        else:
            raise ValueError(f"could not retrieve the document and the boxes errors: {res_document.status_code}, {res_boxes.status_code}")
    
    return document_image_buffer, boxes


def publish_sample_image(image_path, sample_id, document_id):
    url = "http://localhost:8000/document_generator/api/sample_documents/"
    files = {
        "image": open(image_path, "rb")
    }
    data = {
        "name": f"sample_{sample_id}",
        "template_document": document_id
    }

    with requests.Session() as session:
        response = session.post(url, files=files, data=data)

    if response.status_code != 201:
        logging.error(f"unable to publish image: sample_{sample_id}, status: {response.status_code}")

    return response.json()["id"]  


def publish_box_labels(boxes_labels, sample_document_id):
    url = "http://localhost:8000/document_generator/api/sample_boxes/create_sample_boxes/"
    data = {
        "sample_document_id": sample_document_id,
        "boxes": boxes_labels
    }

    with requests.Session() as session:
        response = session.post(url, json=data)

    if response.status_code != 201:
        logging.error(f"unable to publish box labels for sample_document_id: {sample_document_id}, status: {response.status_code}")
