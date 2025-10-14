## Evaluation

### VQAv2
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir vqav2 && cd vqav2`, then download and extract [test2015](http://images.cocodataset.org/zips/test2015.zip) to `./vqav2`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh "MODEL_PATH" "MODEL_NAME"
```
Please submit the results under the `eval_dataset/vqav2/answers_upload` to the [vqav2_evaluation_server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

---

### GQA
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir gqa && cd gqa`, then download the `Questions` and `Images` on [`GQA`](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) and the [Evaluation Script](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) to `./gqa`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh "MODEL_PATH" "MODEL_NAME"
```

---

### Vizwiz
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir vizwiz && cd vizwiz`, then download [Annotations.zip](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and [test.zip](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip). Finally, extract `Annotations/test.json` and `test.zip` to `./vizwiz/test`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vizwiz.sh "MODEL_PATH" "MODEL_NAME"
```
Please submit the results under the `eval_dataset/vizwiz/answers_upload` to the [vizwiz_evaluation_server](https://eval.ai/web/challenges/challenge-page/2185/submission).

---

### SciQA
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir scienceqa && cd scienceqa`. Subsequently, you can download `pid_splits.json`, `problems.json` and `test.zip` on [scienceqa](https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw), and then extract `test.zip` to `./scienceqa/images`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "MODEL_PATH" "MODEL_NAME"
```

---


### TextVQA
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir textvqa && cd textvqa`, then download [TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) to `./textvqa`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh "MODEL_PATH" "MODEL_NAME"
```

---


### MME
1. Prepare the evaluation images as following:

* cd ./eval_dataset/ && mkdir MME && cd MME
* Download the data following the official instructions [Download MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation#:~:text=The%20benchmark%20dataset%20is%20collected%20by%20Xiamen%20University%20for%20academic%20research%20only.%20You%20can%20email%20yongdongluo%40stu.xmu.edu.cn%20to%20obtain%20the%20dataset%2C%20according%20to%20the%20following%20requirement.)
* Downloaded images to `MME/MME_Benchmark_release_version`
* Put the [eval_tool](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/tools/eval_tool.zip) under the `MME/eval_tool`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh "MODEL_PATH" "MODEL_NAME"
```

---


### MMB
1. Prepare the evaluation images as following:

`cd ./eval_dataset/ && mkdir mmbench && cd mmbench`, then download [MMBench_DEV_EN_legacy.tsv](http://opencompass.openxlab.space/utils/MMBench/MMBench_DEV_EN_legacy.tsv) to `mmbench`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/eval/mmbench.sh "MODEL_PATH" "MODEL_NAME" 
```
Please submit the results under the `eval_dataset/mmbench/answers_upload/MMBench_DEV_EN_legacy` to the [mmbench-submission](https://mmbench.opencompass.org.cn/mmbench-submission).

---

### MMBenchCN
1. Prepare the evaluation images as following: 

`cd ./eval_dataset/ && mkdir mmbench_cn && cd mmbench_cn` and then download [MMBench_DEV_CN_legacy.tsv](http://opencompass.openxlab.space/utils/MMBench/MMBench_DEV_CN_legacy.tsv) to `mmbench_cn`

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "MODEL_PATH" "MODEL_NAME"
```
Please submit the results under the `eval_dataset/mmbench_cn/answers_upload/MMBench_DEV_CN_legacy` to the [mmbench-submission](https://mmbench.opencompass.org.cn/mmbench-submission).

---


### POPE
1. Prepare the evaluation images as following: 

`cd ./eval_dataset/ && mkdir pope && cd pope`, then download [json_files](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) to `pope/coco`. In addition, you should download and extract [val2014.zip](http://images.cocodataset.org/zips/val2014.zip) to `pope/val2014`


2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh "MODEL_PATH" "MODEL_NAME"
```

---


### MMMU
1. Prepare the evaluation images as following: 

* cd ./eval_dataset/ && mkdir MMMU && cd MMMU
* Download and Extract [MMMU.zip](https://drive.google.com/file/d/1TJszQ23X-7TeMYDA7hVKpoHy9yo-lsc5/view?usp=sharing)
* Please change `sample["img_path"]` to `./MMMU/all_images` in `eval/download_images.py`，and download images as following:
```bash
python eval/download_images.py
```

2. Evaluate as following. Please change the `MODEL_PATH` and `MODEL_NAME`.
```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/eval/mmmu.sh "MODEL_PATH" "MODEL_NAME"
```

---
The srtucture of the evaluation dataset will be:

```plaintext
├── gqa
│   ├── answers
│   ├── challenge_all_questions.json
│   ├── challenge_balanced_questions.json
│   ├── eval
│   ├── images
│   ├── llava_gqa_testdev_balanced.jsonl
│   ├── readme.txt
│   ├── submission_all_questions.json
│   ├── test_all_questions.json
│   ├── test_balanced_questions.json
│   ├── testdev_all_questions.json
│   ├── testdev_balanced_predictions.json
│   ├── testdev_balanced_questions.json
│   ├── train_all_questions
│   ├── train_balanced_questions.json
│   ├── train_sceneGraphs.json
│   ├── val_all_questions.json
│   ├── val_balanced_questions.json
│   └── val_sceneGraphs.json
├── mmbench
│   ├── answers
│   ├── answers_upload
│   ├── MMBench_DEV_EN_legacy.tsv
├── mmbench_cn
│   ├── answers
│   ├── answers_upload
│   ├── MMBench_DEV_CN_legacy.tsv
├── MME
│   ├── answers
│   ├── convert_answer_to_mme.py
│   ├── eval_tool
│   ├── llava_mme.jsonl
│   └── MME_Benchmark_release_version
├── MMMU
│   ├── all_images
│   ├── anns_for_eval.json
│   ├── answers
│   └── eval
├── pope
│   ├── answers
│   ├── coco
│   ├── llava_pope_test.jsonl
│   └── val2014
├── scienceqa
│   ├── answers
│   ├── images
│   ├── llava_test_CQM-A.json
│   ├── pid_splits.json
│   └── problems.json
├── textvqa
│   ├── answers
│   ├── TextVQA_0.5.1_val.json
│   ├── train_images
│   └── train_images.tar.gz
├── vizwiz
│   ├── Annotations.zip
│   ├── answers
│   ├── answers_upload
│   ├── llava_test.jsonl
│   ├── test
│   ├── train.json
│   └── val.json
└── vqav2
    ├── answers
    ├── answers_upload
    ├── llava_vqav2_mscoco_test2015.jsonl
    ├── llava_vqav2_mscoco_test-dev2015.jsonl
    └── test2015
```