import json
import math
import os
import pandas as pd
from os.path import exists


def approximateTemplateBoxNumbers(file):
    memes = json.load(file)
    memes_nb = len(memes)
    box_count = 0

    for meme in memes:
        box_count += len(meme['boxes'])

    box_per_meme_average = round(box_count / memes_nb)

    return box_per_meme_average


def getTemplatesBoxCount(file_path):
    work_dir = os.getcwd()

    os.chdir(os.path.join(work_dir, file_path))

    new_work_dir = os.getcwd()

    box_count_per_template_list = []

    for file in os.listdir():
        if file.endswith(".json"):
            box_count_per_template_list.append(approximateTemplateBoxNumbers(open(os.path.join(new_work_dir, file))))

    os.chdir(work_dir)

    return box_count_per_template_list


def getDataset(file_path):
    templates_box_count = getTemplatesBoxCount(file_path)

    work_dir = os.getcwd()

    os.chdir(os.path.join(work_dir, file_path))

    new_work_dir = os.getcwd()

    x = 0
    df = None

    for file in os.listdir():
        if file.endswith(".json"):
            file_name = getFileName(file)
            if df is None:
                df = getMemesTemplateDF(open(os.path.join(new_work_dir, file)), templates_box_count[x], file_name)
            else:
                df = pd.concat(
                    [df, getMemesTemplateDF(open(os.path.join(new_work_dir, file)), templates_box_count[x], file_name)],
                    ignore_index=True)
            x += 1

    os.chdir(work_dir)
    df.to_csv("memes.csv")

    return df


def getMemesTemplateDF(file, box_count, template_names):
    memes = json.load(file)

    authors = []
    url = []
    views = []
    templates = []
    titles = []
    contents = []
    votes = []
    boxes = []

    for meme in memes:
        if len(meme["boxes"]) == box_count:
            templates.append(template_names)
            url.append(meme['url'])
            votes.append(int(meme["metadata"]["img-votes"]))
            views.append(convertViews(meme["metadata"]["views"]))
            authors.append(meme["metadata"]["author"])
            titles.append(meme["metadata"]["title"])
            contents.append(formatContent(meme["boxes"]))
            boxes.append(box_count)

    data = list(zip(titles, templates, authors, contents, boxes, url, votes, views))
    df = pd.DataFrame(data,
                      columns=["title", "template", "author", "content", "number_of_boxes", "url", "votes", "views"])

    return df


def convertViews(views_str):
    try:
        views_str = views_str.replace(",", ".")
        number_of_decimal = len(views_str.split(".")[1].rstrip("0"))
        converted_views = int(math.pow(10, number_of_decimal) * float(views_str))
    except:
        converted_views = int(views_str)

    return converted_views


def formatContent(meme_boxes):
    formated_content = ""

    number_of_boxes = len(meme_boxes)

    for x in range(number_of_boxes):
        formated_content += meme_boxes[x]
        if x < (number_of_boxes - 1):
            formated_content += "\n"

    return formated_content


def getFileName(file_name_with_extension):
    return file_name_with_extension[0:len(file_name_with_extension) - 5]


def isDatasetExist():
    dataset_file_existance = exists("memes.csv")

    return dataset_file_existance

def importDataset():
    return pd.read_csv("memes.csv")
