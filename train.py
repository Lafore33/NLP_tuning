import spacy
import matplotlib.pyplot as plt
from spacy.training import Example
import numpy as np
import json
from random import shuffle


def train_fn(model, data, epochs, batch_size=5):
    optimizer = model.resume_training()
    loss_history = []
    size = len(data)

    for epoch in range(epochs):
        shuffle(data)
        batches = [data[i:i + batch_size] for i in range(0, size, batch_size)]
        losses = {}
        cur_sum_loss = 0

        for batch in batches:
            examples = []
            for annotation in batch:
                doc = model.make_doc(annotation["text"])
                example = Example.from_dict(doc, {"entities": annotation["entities"]})
                examples.append(example)

            losses = model.update(examples, drop=0.1, losses=losses, sgd=optimizer)
            cur_sum_loss += losses["ner"]

        avg_loss = cur_sum_loss / size
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}: {avg_loss}")

        if epoch % 10 == 0:
            show_losses(loss_history)


def show_losses(train_loss_hist):
    plt.figure()
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.title("Train Loss")
    plt.yscale("log")
    plt.grid()
    plt.show()


def load_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    train_data = []
    for pair in data["annotations"]:
        ent = []
        for e in pair["entities"]:
            ent.append(tuple(e.values()))
        train_data.append({"text": pair["text"], "entities": ent})

    return train_data


def test(model, tuned_model_path, data):
    tuned_nlp = spacy.load(tuned_model_path)
    doc = model(data)
    tuned_doc = tuned_nlp(data)

    for ent in tuned_doc.ents:
        print(ent.text, ent.label_)
    print("---")
    for ent in doc.ents:
        print(ent.text, ent.label_)


def main():
    nlp = spacy.load("en_core_web_lg")
    train_mode = False
    data_file = "data.json"
    model_folder = "tuned_model"

    if train_mode:

        data = load_data(data_file)

        nlp.disable_pipes(*[pipe_name for pipe_name in nlp.pipe_names if pipe_name not in ["ner"]])
        nlp.config["training"]["frozen_components"] = ["tok2vec"]

        train_fn(nlp, data, epochs=100)
        nlp.to_disk(model_folder)

    else:
        sentence = "Arsenal is in london, I am in italy talking spanish at 8am with 30 other people"
        test(nlp, model_folder, sentence)


if __name__ == "__main__":
    main()
