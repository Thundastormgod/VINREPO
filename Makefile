.PHONY: get-grayscale-data get-rgb-data clean-data vin-ocr-install vin-ocr-fetch

DATA_DIR := data
TRAIN_DIR := $(DATA_DIR)/train
TEST_DIR := $(DATA_DIR)/test

RGB_TRAIN_DIR := $(DATA_DIR)/rgb-train
RGB_TEST_DIR := $(DATA_DIR)/rgb-test

TMP_DIR := $(DATA_DIR)/.simpsons-tmp
REPO := https://github.com/alvarobartt/simpsons-mnist.git

# Grayscale
get-grayscale-data: $(TRAIN_DIR) $(TEST_DIR) clean-data

$(TMP_DIR):
	rm -rf $(TMP_DIR)
	git clone --filter=blob:none --sparse $(REPO) $(TMP_DIR)

$(TRAIN_DIR): $(TMP_DIR)
	cd $(TMP_DIR) && git sparse-checkout set dataset/grayscale/train dataset/grayscale/test
	mkdir -p $(DATA_DIR)
	cp -R $(TMP_DIR)/dataset/grayscale/train $(TRAIN_DIR)

$(TEST_DIR): $(TMP_DIR)
	cd $(TMP_DIR) && git sparse-checkout set dataset/grayscale/train dataset/grayscale/test
	mkdir -p $(DATA_DIR)
	cp -R $(TMP_DIR)/dataset/grayscale/test $(TEST_DIR)

# RGB
get-rgb-data: $(RGB_TRAIN_DIR) $(RGB_TEST_DIR) clean-data

$(RGB_TRAIN_DIR): $(TMP_DIR)
	cd $(TMP_DIR) && git sparse-checkout set dataset/rgb/train dataset/rgb/test
	mkdir -p $(DATA_DIR)
	cp -R $(TMP_DIR)/dataset/rgb/train $(RGB_TRAIN_DIR)

$(RGB_TEST_DIR): $(TMP_DIR)
	cd $(TMP_DIR) && git sparse-checkout set dataset/rgb/train dataset/rgb/test
	mkdir -p $(DATA_DIR)
	cp -R $(TMP_DIR)/dataset/rgb/test $(RGB_TEST_DIR)

clean-data:
	rm -rf $(TMP_DIR)

vin-ocr-install:
	pip install -r vin_ocr/requirements.txt

vin-ocr-fetch:
	python vin_ocr/tools/dagshub_fetch.py \
	  --repo-url https://dagshub.com/Thundastormgod/jlr-vin-ocr \
	  --repo-data-path data \
	  --output-dir vin_ocr_data/raw \
	  --clone-dir vin_ocr_data/.dagshub_repo \
	  --clean
