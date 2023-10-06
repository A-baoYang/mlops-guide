## BentoML

以最簡單的機器學習模型為示範

1. 訓練模型

```bash
python train.py
```

2. If Model API: 撰寫 Bentoml API 服務檔；接著運行

```bash
bentoml serve ./service.py:svc --reload
```

3. If docker image: 撰寫 `bentofile.yml` 用來定義 image 環境所需套件、要拉近哪些檔案；接著運行

```bash
bentoml build -f bentofile.yml
```
接著再運行，即可獲得 image
```bash
bentoml containerize model_name:image_tag
```
- image_tag: 例如 `latest`

