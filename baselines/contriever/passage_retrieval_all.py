# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
import jsonlines
from pathlib import Path

import numpy as np
import torch
import transformers
import tqdm

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text


import pickle
import torch 
import io

import re

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_first_4_digit_number(text):
    pattern = r"\b\d{4}\b"
    match = re.search(pattern, text)
    
    if match:
        return match.group()
    else:
        return None


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())


                for idx, q in enumerate(batch_question):
                    if q == '':
                        embeddings[0][idx] = 0
                
                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()

def embed_queries_clip(queries, model):
    
    embeddings = model.encode([query[:100] for query in queries], convert_to_tensor=True, show_progress_bar=False)

    
    # zero out empty queries
    for idx, q in enumerate(queries):
        if q == '':
            embeddings[idx] = 0

    # move to cpu

    print(f"Clip embedding shape: {embeddings.size()}")
    return embeddings.cpu().numpy()



def index_encoded_data(index, blurb_embedding_files, title_embedding_files, genre_embedding_files, author_embedding_files, indexing_batch_size):

    allids = []
    allembeddings = np.array([])
    for i, (blurb_file_path, title_file_path, genre_file_path, author_file_path) in enumerate(zip(blurb_embedding_files, title_embedding_files, genre_embedding_files, author_embedding_files)):
        print(f"Loading file {blurb_file_path}")
        with open(blurb_file_path, "rb") as fin:
            ids, blurb_embeddings = pickle.load(fin)

        print(f"Loading file {title_file_path}")
        with open(title_file_path, "rb") as fin:
            _, title_embeddings = pickle.load(fin)
        
        print(f"Loading file {genre_file_path}")
        with open(genre_file_path, "rb") as fin:
            _, genre_embeddings = pickle.load(fin)
        
        print(f"Loading file {author_file_path}")
        with open(author_file_path, "rb") as fin:
            _, author_embeddings = pickle.load(fin)


        id_to_parsed_dates = {}
        with jsonlines.open("/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_books.jsonl", "r") as reader:
            for obj in reader:
                if obj['parsed_dates']:
                    year = obj['parsed_dates'][0][:4]
                    id_to_parsed_dates[obj['id']] = year

        # make numpy array of size (len(ids), 48) with parsed dates
        date_embeddings = np.zeros((len(ids), 48))
        for i, id in enumerate(ids):
            year = id_to_parsed_dates.get(id, 1975)
            date_embeddings[i][min(max(int(year)-1975, 0), 47):] = 1

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open('/home/kevinlin/whatsthatbook/data/wtb_filename_to_image_embeddings_finetuned.pickle', 'rb') as handle:
            wtb_filename_to_image_embeddings_finetuned =CPU_Unpickler(handle).load()

        ids_to_image_embeddings = {}
        for key, value in wtb_filename_to_image_embeddings_finetuned.items():
            curr_id = key.split('/')[-1]
            embedding = value.detach().cpu().numpy()
            # embedding  = np.resize(embedding, (768, 1))
            ids_to_image_embeddings[curr_id] = embedding
        
        image_embeddings = np.stack([ids_to_image_embeddings[book_id] for book_id in ids])


        embeddings = np.concatenate((blurb_embeddings,
                                     title_embeddings,
                                     genre_embeddings,
                                     author_embeddings,
                                     image_embeddings,
                                     date_embeddings), axis=1)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits, message


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    blurb_model, tokenizer, _ = src.contriever.load_retriever(args.blurb_model_name_or_path)
    blurb_model.eval()
    blurb_model = blurb_model.cuda()
    if not args.no_fp16:
        blurb_model = blurb_model.half()

    title_model, tokenizer, _ = src.contriever.load_retriever(args.title_model_name_or_path)
    title_model.eval()
    title_model = title_model.cuda()
    if not args.no_fp16:
        title_model = title_model.half()

    genre_model, tokenizer, _ = src.contriever.load_retriever(args.genre_model_name_or_path)
    genre_model.eval()
    genre_model = genre_model.cuda()
    if not args.no_fp16:
        genre_model = genre_model.half()

    author_model, tokenizer, _ = src.contriever.load_retriever(args.author_model_name_or_path)
    author_model.eval()
    author_model = author_model.cuda()
    if not args.no_fp16:
        author_model = author_model.half()
    import clip
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel

    model = SentenceTransformer('clip-ViT-B-32')
    model.eval()
    model = model.cuda()


    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    checkpoint = torch.load(args.clip_model_name_or_path, map_location="cpu")


    clip_model.load_state_dict(checkpoint['model_state_dict'])
    model[0].model = clip_model


    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    # index all passages
    blurb_input_paths= glob.glob(args.blurb_passages_embeddings)
    blurb_input_paths = sorted(blurb_input_paths)
    embeddings_dir = os.path.dirname(blurb_input_paths[0])
    blurb_index_path = os.path.join(embeddings_dir, "index.faiss")

    # same thing for titles
    title_input_paths= glob.glob(args.title_passages_embeddings)
    title_input_paths = sorted(title_input_paths)
    embeddings_dir = os.path.dirname(title_input_paths[0])
    title_index_path = os.path.join(embeddings_dir, "index.faiss")

    # same thing for titles
    genre_input_paths= glob.glob(args.genre_passages_embeddings)
    genre_input_paths = sorted(genre_input_paths)
    embeddings_dir = os.path.dirname(genre_input_paths[0])
    genre_index_path = os.path.join(embeddings_dir, "index.faiss")

    # same thing for titles
    author_input_paths= glob.glob(args.author_passages_embeddings)
    author_input_paths = sorted(author_input_paths)
    embeddings_dir = os.path.dirname(author_input_paths[0])
    author_index_path = os.path.join(embeddings_dir, "index.faiss")


    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        #print(f"Indexing passages from files {[text_input_paths, title_input_paths]}")
        start_time_indexing = time.time()
        index_encoded_data(index, blurb_input_paths, title_input_paths, genre_input_paths, author_input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = src.data.load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    data_paths = glob.glob(args.data)
    alldata = []
    for path in data_paths:

        id_to_title_clue = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.title_clues.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                id_to_title_clue[obj['title']['book_id']] = obj      

        id_to_blurb_clue = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts_plot_extract_clues.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                id_to_blurb_clue[obj['title']['book_id']] = obj

        id_to_genre_clue = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.genre_clues.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                id_to_genre_clue[obj['title']['book_id']] = obj

        id_to_author_clue = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.author_clues.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                id_to_author_clue[obj['title']['book_id']] = obj                

        id_to_cover_clue = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.cover_clues.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                id_to_cover_clue[obj['title']['book_id']] = obj

        description_to_book = {}
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_books.jsonl') as reader:
            for obj in tqdm.tqdm(reader):
                description_to_book[obj['description']] = obj
        print(f'description to book {len(description_to_book)}') 

        data = load_data(path)
        output_path = os.path.join(args.output_dir, os.path.basename(path))

        queries = [ex["question"] for ex in data]

        blurb_queries = []
        for ex in data:
            book = id_to_blurb_clue.get(description_to_book[ex["answers"][0]]['id'])
            if book:
                clue = book['comments'][0]['comment_text']
                if "N/A" not in clue:
                    blurb_queries.append(clue)
                else:
                    blurb_queries.append("")
            else:
                    blurb_queries.append("")


        title_queries = []
        for ex in data:
            book = id_to_title_clue.get(description_to_book[ex["answers"][0]]['id'])
            if book:
                clue = book['comments'][0]['comment_text']
                if "N/A" not in clue:
                    title_queries.append(clue)
                else:
                    title_queries.append("")
            else:
                    title_queries.append("")


        genre_queries = []
        for ex in data:
            book = id_to_genre_clue.get(description_to_book[ex["answers"][0]]['id'])
            if book:
                clue = book['comments'][0]['comment_text']
                if "N/A" not in clue:
                    genre_queries.append(clue)
                else:
                    genre_queries.append("")
            else:
                    genre_queries.append("")

        author_queries = []
        for ex in data:
            book = id_to_author_clue.get(description_to_book[ex["answers"][0]]['id'])
            if book:
                clue = book['comments'][0]['comment_text']
                if "N/A" not in clue:
                    author_queries.append(clue)
                else:
                    author_queries.append("")
            else:
                    author_queries.append("")

        cover_queries = []
        for ex in data:
            book = id_to_cover_clue.get(description_to_book[ex["answers"][0]]['id'])
            if book:
                clue = book['comments'][0]['comment_text']
                if "N/A" not in clue:
                    cover_queries.append(clue)
                else:
                    cover_queries.append("")
            else:
                    cover_queries.append("")

        blurb_embedding = embed_queries(args, blurb_queries, blurb_model, tokenizer)
        title_embedding = embed_queries(args, title_queries, title_model, tokenizer)
        genre_embedding = embed_queries(args, genre_queries, genre_model, tokenizer)
        author_embedding = embed_queries(args, author_queries, author_model, tokenizer)

        clip_embedding = embed_queries_clip(cover_queries, model)

        query_to_year = {}
        
        with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.test.jsonl', 'r') as reader:
            with jsonlines.open('/home/kevinlin/whatsthatbook/data/2022-05-30_14441_gold_posts.test.dates.relative.jsonl', 'r') as dates_reader:
                for obj, date_obj in zip(reader, dates_reader):
                    query = obj['comments'][0]['comment_text']
                    output = date_obj['comments'][0]['comment_text']
                    extracted_year = get_first_4_digit_number(output)
                    if extracted_year == None:
                        extracted_year = 2023
                    query_to_year[query] = max(1975, min(int(extracted_year) + 15, 2023))

        extracted_years = []
        for query in queries:
            extracted_years.append(query_to_year.get(query, 2023))
        date_embedding = np.zeros((len(extracted_years), 48))

        for idx, year in enumerate(extracted_years):
            date_embedding[idx][max(year - 1975 - 1, 0)] = 1


        config_weight_files = glob.glob(args.configs_glob)

        best_top1 = 0
        best_config = None
        best_message = None
        for config_weight_file in tqdm.tqdm(config_weight_files):
            with open(config_weight_file) as f:
                config = json.load(f)

            weighted_blurb_embedding = blurb_embedding * config['blurb_weight']
            weighted_clip_embedding = clip_embedding * config['image_weight']
            weighted_title_embedding = title_embedding * config['title_weight']
            weighted_genre_embedding = genre_embedding * config['genre_weight']
            weighted_author_embedding = author_embedding * config['author_weight']
            weighted_date_embedding = date_embedding * config['date_weight']


            joint_questions_embedding = np.concatenate([weighted_blurb_embedding,
                                                        weighted_title_embedding,
                                                        weighted_genre_embedding,
                                                        weighted_author_embedding,
                                                        weighted_clip_embedding,
                                                        weighted_date_embedding], axis=1)
            
            # save to weight to output paths name
            # output_path = os.path.join(args.output_dir, os.path.basename(path).replace(".jsonl", f".clip_weight_{clip_weight}.jsonl"))

            # get top k results
            start_time_retrieval = time.time()

            top_ids_and_scores = index.search_knn(joint_questions_embedding, args.n_docs)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

            add_passages(data, passage_id_map, top_ids_and_scores)
            hasanswer, message  = validate(data, args.validation_workers)
            top1 = sum([example_hasanswer[0] for example_hasanswer in hasanswer]) / len(hasanswer)
            if top1 > best_top1:
                best_top1 = top1
                best_config = config_weight_file
                best_message = message
            print(config_weight_file)
            print("current best message is: ", best_message)
            print("curren best config is: ", best_config)

            add_hasanswer(data, hasanswer)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fout:
                for ex in data:
                    json.dump(ex, fout, ensure_ascii=False)
                    fout.write("\n")
            print(f"Saved results to {output_path}")

        print(f"Best config: {best_config} best top1: {best_top1}, message: {best_message}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--text_passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--blurb_passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--title_passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--genre_passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--author_passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--text_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    
    parser.add_argument(
        "--blurb_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )


    parser.add_argument(
        "--title_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )

    parser.add_argument(
        "--genre_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )

    parser.add_argument(
        "--author_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )


    parser.add_argument(
        "--clip_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )

    parser.add_argument(
        "--configs_glob", type=str, help="path to directory containing model weights and config file"
    )


    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
