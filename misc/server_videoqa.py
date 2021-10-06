#!/usr/bin/env python
import os
import json
import torch
import torch.nn.functional as F
import pickle
import random
import urllib
import urllib.request
import cherrypy
from transformers import DistilBertTokenizer
from model.multimodal_transformer import MMT_VideoQA
from util import compute_a2v, get_mask
from args import get_args
from global_parameters import (
    SERVER_HTML_PATH,
    SERVER_FEATURE_PATH,
)  # to be defined in this file


class Server(object):
    def __init__(
        self,
        vqa_model,
        vqa_model2,
        model_ckpt,
        model_ckpt2,
        video_features_path,
        a2v,
        id2a,
        T,
        Q,
        default_data,
        max_videos,
    ):
        """
        :param vqa_model: first model used for the demo
        :param vqa_model2: second model used for the demo
        :param model_ckpt: path to weights for the first model
        :param model_ckpt2: path to weights for the second model
        :param video_features_path: path to the features corresponding to the videos used in the demo
        :param a2v: map answer to tokens for all answers in a given answer dictionary
        :param id2a: map index to answer
        :param T: maximum number of video features
        :param Q: maximum number of tokens in the question
        :param default_data: map video_id to question, start, end
        :param max_videos: maximum number of videos in the demo
        """
        self.video_features = torch.load(video_features_path)

        # load weights for the first model on CPU
        self.vqa_model = vqa_model
        weights = torch.load(model_ckpt, map_location=torch.device("cpu"))
        weights = {x.split("module.")[1]: weights[x] for x in weights}
        self.vqa_model.load_state_dict(weights)
        self.vqa_model.eval()
        self.vqa_model._compute_answer_embedding(a2v)

        # load weights for the second model on CPU
        self.vqa_model2 = vqa_model2
        weights2 = torch.load(model_ckpt2, map_location=torch.device("cpu"))
        weights2 = {x.split("module.")[1]: weights2[x] for x in weights2}
        self.vqa_model2.load_state_dict(weights2)
        self.vqa_model2.eval()
        self.vqa_model2._compute_answer_embedding(a2v)

        self.all_video_ids = list(self.video_features.keys())[:max_videos]
        self.id2a = id2a
        self.T = T
        self.Q = Q
        self.default_data = default_data
        self.max_videos = max_videos

    @cherrypy.expose
    def index(self):
        index_html = '<head><link rel="icon" href="https://antoyang.github.io/img/favicon.ico" type="image/x-icon"/>'
        index_html += '<link href="https://antoyang.github.io/css/bootstrap.min.css" rel="stylesheet"></head>'
        index_html += "<center><h1> <a href='https://antoyang.github.io/just-ask.html'> Just Ask </a> VideoQA Demo </h1></center>"
        index_html += "<center><h2> Choose a video for which you want to ask a question </h2></center>"
        index_html += "<center><h3> Default question, start and end timestamps are from the iVQA test set annotations. Nothing is pre-computed for these videos. </h3></center><br>"
        index_html += '<div class="container">'  # grid of videos
        for i, vid in enumerate(self.all_video_ids):
            url = "https://www.youtube.com/oembed"
            params = {
                "format": "json",
                "url": "https://www.youtube.com/watch?v=%s" % vid,
            }
            query_string = urllib.parse.urlencode(params)
            url = url + "?" + query_string
            try:
                with urllib.request.urlopen(
                    url
                ) as response:  # get thumbnail and title from YouTube
                    response_text = response.read()
                    data = json.loads(response_text.decode())
                    # pprint.pprint(data)
                    title = data["title"]
                    thumbnail_url = data["thumbnail_url"]
            except:  # if the video is deleted: json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
                title = "Unavailable Video"
                thumbnail_url = "https://images.drivereasy.com/wp-content/uploads/2017/10/this-video-is-not-available-1.jpg"
            if i % 4 == 0:  # 4 videos per row
                index_html += '<div class="row">'
            index_html += '<div class="col-md-3 col-sm-12"><center><a href="vqa?video_id={}"><img src={} height="180" width="240"></img></a><br>'.format(
                vid, thumbnail_url
            )
            index_html += '<a href="vqa?video_id={}">{}</a></center></div>'.format(
                vid, title
            )
            if (i % 4 == 3) or (
                i == min(len(self.all_video_ids), self.max_videos) - 1
            ):  # end of row
                index_html += "</div><br><br>"
        index_html += "</div>"

        index_html += "<center><a href='reload' class='btn btn-primary btn-lg active'>More videos!</a></center><br>"
        index_html += "<center><h2> Built by <a href='https://antoyang.github.io/'> Antoine Yang </a> </h2> </center><br>"
        return index_html

    @cherrypy.expose
    def vqa(self, video_id, start=0, end=5, question="", model="finetuned"):
        if video_id not in self.video_features:
            return (
                f'Video {video_id} is not available, <a href="/">go back to index</a>.'
            )
        html_path = SERVER_HTML_PATH
        with open(html_path, "r") as f:
            html = f.read()
        if not str(start).isdigit():
            return 'Start time (in seconds) must be a positive integer, <a href="/">go back to index</a>.'
        if not str(end).isdigit():
            return 'End time (in seconds) must be a positive integer, <a href="/">go back to index</a>.'
        if not question:  # put default data
            flag = False
            start = self.default_data[video_id]["start"]
            end = self.default_data[video_id]["end"]
            question = self.default_data[video_id]["question"]
        else:
            flag = True  # a question is asked
        html = html.format(video_id, start, end, video_id, start, end, question)
        feature = self.video_features[video_id][int(start) : int(end) + 1]
        if len(feature) == 0:
            return f'Features are not available for video {video_id} between start {start} seconds and {end} seconds, <a href="/">go back to index</a>.'
        if flag:
            # prepare padded features and tokens, masks
            video_len = torch.Tensor([len(feature)])
            if len(feature) < self.vqa_model.T:
                feature = torch.cat(
                    [
                        feature,
                        torch.zeros(self.vqa_model.T - len(feature), feature.size(1)),
                    ],
                    dim=0,
                )
            else:
                sampled = []
                for j in range(self.vqa_model.T):
                    sampled.append(feature[(j * len(feature)) // self.vqa_model.T])
                feature = torch.stack(sampled)

            feature = feature.unsqueeze(0)
            video_mask = get_mask(video_len, self.vqa_model.Q)

            tokens = torch.tensor(
                self.vqa_model.bert.bert_tokenizer.encode(
                    question,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=self.vqa_model.Q,
                    truncation=True,
                ),
                dtype=torch.long,
            ).unsqueeze(0)
            question_mask = tokens > 0

            with torch.no_grad():  # forward
                if (
                    model == "zeroshot"
                ):  # assumes that the first model is the zeroshot one
                    predicts = self.vqa_model(
                        feature,
                        question=tokens,
                        video_mask=video_mask,
                        text_mask=question_mask,
                    )
                elif model == "finetuned":
                    predicts = self.vqa_model2(
                        feature,
                        question=tokens,
                        video_mask=video_mask,
                        text_mask=question_mask,
                    )
                else:
                    raise NotImplementedError
                predicts = F.softmax(predicts, dim=1)
                topk = torch.topk(predicts, dim=1, k=5)  # top 5 answers
                topk_txt = [
                    [self.id2a[x.item()] for x in y] for y in topk.indices.cpu()
                ]
                topk_scores = [[x * 100 for x in y] for y in topk.values.cpu()]
            progress_bar = ""
            for i in range(5):  # plot answer logits with a nice progress bar
                progress_bar += f'<div class="row"><div class="col-md-3" style="height: 5%;"><h3 style="color: #428bca !important;" class="center">{topk_txt[0][i]}</h3></div>'
                progress_bar += f'<div class="col-md-9" style="height: 5%;"><div class="progress" style="margin-top: 20px !important;"><div class="progress-bar" style="color: black; width: {topk_scores[0][i]}%;" width: {topk_scores[0][i]}%;" role="progressbar" aria-valuenow="{topk_scores[0][i]}" aria-valuemin="0" aria-valuemax="1">{topk_scores[0][i]:.2f}%</div></div></div></div>'
            html += '<div class="col-sm-offset-2 col-sm-8"> <b> Question input </b>: {} <br> <b> <br> Top 5 answers ({} model) </b>: {} </div></div>'.format(
                question, model, progress_bar
            )

        return html + "</div><br><br></body></html>"

    @cherrypy.expose
    def reload(self):  # same as index after a randomizing the videos
        self.all_video_ids = random.sample(
            list(self.video_features.keys()), self.max_videos
        )

        index_html = '<head><link rel="icon" href="https://antoyang.github.io/img/favicon.ico" type="image/x-icon"/>'
        index_html += '<link href="https://antoyang.github.io/css/bootstrap.min.css" rel="stylesheet"></head>'  # https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css
        index_html += "<center><h1> <a href='https://antoyang.github.io/just-ask.html'> Just Ask </a> VideoQA Demo </h1></center>"
        index_html += "<center><h2> Choose a video for which you want to ask a question </h2></center>"
        index_html += "<center><h3> Default question, start and end timestamps are from the iVQA test set annotations. Nothing is pre-computed for these videos. </h3></center><br>"
        index_html += '<div class="container">'
        for i, vid in enumerate(self.all_video_ids):
            url = "https://www.youtube.com/oembed"
            params = {
                "format": "json",
                "url": "https://www.youtube.com/watch?v=%s" % vid,
            }
            query_string = urllib.parse.urlencode(params)
            url = url + "?" + query_string
            try:
                with urllib.request.urlopen(url) as response:
                    response_text = response.read()
                    data = json.loads(response_text.decode())
                    title = data["title"]
                    thumbnail_url = data["thumbnail_url"]
            except:
                title = "Unavailable Video"
                thumbnail_url = "https://images.drivereasy.com/wp-content/uploads/2017/10/this-video-is-not-available-1.jpg"
            if i % 4 == 0:
                index_html += '<div class="row">'
            index_html += '<div class="col-md-3 col-sm-12"><center><a href="vqa?video_id={}"><img src={} height="180" width="240"></img></a><br>'.format(
                vid, thumbnail_url
            )
            index_html += '<a href="vqa?video_id={}">{}</a></center></div>'.format(
                vid, title
            )
            if (i % 4 == 3) or (i == min(len(self.all_video_ids), self.max_videos) - 1):
                index_html += "</div><br><br>"
        index_html += "</div>"
        index_html += "<center><a href='reload' class='btn btn-primary btn-lg active'>More videos!</a></center><br>"
        index_html += "<center><h2> Built by <a href='https://antoyang.github.io/'> Antoine Yang </a> </h2> </center><br>"
        return index_html


def run():
    args = get_args()
    port = args.port
    cherrypy.config.update({"server.socket_port": port})
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    conf = {
        "/": {
            "tools.sessions.on": True,
            "tools.staticdir.root": os.path.abspath(os.getcwd()),
        },
        "/js": {"tools.staticdir.on": True, "tools.staticdir.dir": "./js"},
    }
    dir_map = {
        "activitynet": "ActivityNet-QA",
        "msrvtt": "MSRVTT-QA",
        "msvd": "MSVD-QA",
        "ivqa": "iVQA",
    }
    feature_path = os.path.join(
        SERVER_FEATURE_PATH, dir_map[args.dataset], "full_s3d_features_test.pth"
    )  # path to S3D features extracted for the full video duration

    default_data = pickle.load(
        open(
            os.path.join(
                SERVER_FEATURE_PATH, dir_map[args.dataset], "default_test.pkl"
            ),
            "rb",
        )
    )  # dictionary mapping video_id to question, start and end extracted from the dataset

    bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    a2id, id2a, a2v = compute_a2v(
        vocab_path=args.vocab_path,
        bert_tokenizer=bert_tokenizer,
        amax_words=args.amax_words,
    )
    a2v = a2v.cpu()
    vqa_model = MMT_VideoQA(
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        baseline=args.baseline,
    )

    vqa_model2 = MMT_VideoQA(
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        baseline=args.baseline,
    )

    print(f"http server is running at port {port}")
    cherrypy.quickstart(
        Server(
            vqa_model,
            vqa_model2,
            args.pretrain_path,
            args.pretrain_path2,
            feature_path,
            a2v,
            id2a,
            args.max_feats,
            args.qmax_words,
            default_data,
            args.nb_examples,
        ),
        "/",
        conf,
    )


if __name__ == "__main__":
    run()
