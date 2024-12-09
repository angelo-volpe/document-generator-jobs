import requests


def get_image_and_boxes(document_id: str):
    with requests.Session() as session:
        res_document = session.get(f"http://localhost:8000/document_generator/documents/{document_id}")
        res_boxes = session.get(f"http://localhost:8000/document_generator/{document_id}/boxes")
    
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


def denormalise_box_coordinates(start_x_norm, start_y_norm, end_x_norm, end_y_norm, doc_width, doc_height):
    start_x = int(start_x_norm * doc_width)
    end_x = int(end_x_norm * doc_width)
    start_y = int(start_y_norm * doc_height)
    end_y = int(end_y_norm * doc_height)
    
    return start_x, start_y, end_x, end_y