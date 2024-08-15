# CraftDroid v.2023

This is a quick patch of CraftDroid to accommodate recent versions of its dependent frameworks/Python libraries, e.g., Appium, Flask, BeautifulSoup, gensim, etc. This patch is tested with the environment below:

* Windows 11 / Python 3.12.0
* node.js v20.9.0 / appium v2.2.1 / uiautomator2@2.34.0
* Android Studio Giraffe | 2022.3.1 Patch 2 / Nexus_5X Emulator w/ API_23 (x86_64 image)

# Prerequisites

* Python 3.12 with the packages in `requirements.txt` installed
* [Appium v2.2.1](https://appium.io/docs/en/2.1/quickstart/install/) with [UiAutomator2 Driver](https://appium.io/docs/en/2.1/quickstart/uiauto2-driver/) installed
* [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
* Android Studio Giraffe | 2022.3.1 Patch 2 / Nexus_5X Emulator w/ API_23 (x86_64 image)
* Download the [subject apks](https://drive.google.com/open?id=1wb9ODzqMfsRCLqU80QF-g_1IrF0r7-vj) and `git clone` this project.

# Getting Started
1. Install subject apps on the emulator; we suggest starting with the apps under a2-todo/ to avoid some network issues of apps
2. Start the emulator; Start appium
3. `python w2v_service.py` first to activate the background web service for similarity query (modify the path to `GoogleNews-vectors-negative300.bin` in the source code)
4. Run Explorer.py with arguments: 
```
python3 Explorer.py ${TRANSFER_ID} ${APPIUM_PORT} ${EMULATOR}
```
TRANSFER_ID is the transfer id, APPIUM_PORT is the port used by Appium-Desktop (4723 by default), EMULATOR is the name of the emulator, e.g., 
```
python3 Explorer.py a21-a22-b21 4723 emulator-5554
```
It will start transferring the test case of `a21-Minimal` to `a22-Clear` List for the b21-Add task function. 

5. The source test cases are under test-repo/[CATEGORY]/[FUNCTIONALITY]/base/[APP_ID].json, e.g., `test-repo/a2/b21/base/a21.json`. The generated test cases for the target app is under generated/[APP_FROM]-/[APP_TO]-[FUNCTIONALITY].json, e.g., `test-repo/a2/b21/generated/a21-a22-b21.json`

We also suggest turning off animations on the emulator to avoid potential interaction issues.

![animation-off](./animation-off.jpg)
# FAQ

## What's inside `sa_info`?

`sa_info` folder contains information from static analysis. There are two main parts for each app in the folder:

1. `res` folder and AndroidManifest.xml. They are generated by [Apktool](https://ibotpeaches.github.io/Apktool/), an APK disassembler.
2. `atm` folder, which contains an initial UI Transition Graph (atm.gv) and the integer IDs used by the apk (constantInfo.csv). These files are generated by Model Extractor. You can find the source code of Model Extractor [here](https://drive.google.com/file/d/1HEFS9_6c5nNKnzBPkWlRdwBiunOHgOs-/view?usp=sharing).

Note that the current implementation of Model Extractor reuses an analysis described in TrimDroid (our ICSE 2016 paper). we are planning to refactor Model Extractor to make it more organized and publicly ready. Also, due to some issues such as the version of Soot or obfuscated commercial apps, Model Extractor may not be able to analyze some apks. However, CraftDroid is able to run without the information in "sa_info” folder; in that case it relies on dynamic analysis only.

## How to generate the json test cases in `test_repo`?

The json files are automatically generated from instrumented test cases written in Python with Appium library. The instrumentation of tests is currently manual. For example, there are examples of an original test and instrumented test for the app `a21` under `test_repo/a2/example_appium_test`. 

When running the instrumented test, depends on the version of BeautifulSoup (the DOM parsing library used in CraftDroid), you may get an error saying "AttributeError: 'NavigableString' object has no attribute 'attrs'". A way to resolve the error is to modify `WidgetUtil.py` as follows:

* line 1: change `from bs4 import BeautifulSoup` to `from bs4 import BeautifulSoup, NavigableString`
* line 91, in the function `get_sibling_text()`: change 
```
if prev_sib and 'text' in prev_sib.attrs and prev_sib['text']:
``` 
to 
```
if prev_sib and not isinstance(prev_sib, NavigableString) and 'text' in prev_sib.attrs and prev_sib['text']:
```

## What's the difference between `base_form` and `base_to` in `Evaluator.py`?

`Evaluator.py` is used to evaluate the test cases generated by CraftDroid in terms of precision, recall, etc.
Let's say we would like to transfer the test `a2/b21/base/a21.json` to the app `a22`. The GUI events in `a2/b21/base/a21.json` (source events) will be loaded with `base_from` option. The correct GUI events to test the same feature on `a22`, i.e., `a2/b21/base/a22.json` (target events), will be loaded with `base_to` option. 

Also, we need a solution file that states the mapping between source and target events, so that we can know whether the events generated by CraftDroid are correct, i.e., identical to the target events. You can find the solution files used in our evaluation [here](https://drive.google.com/open?id=14J-4QLQjwN4_lhR87BRRPcxBzE_eeU4I).

## What's the meaning of the column in file `config.csv`?  

They are options that we can determine to turn on or off when transferring tests and computing similarity scores for a pair of GUI widgets.

* `use_stopwords`: to remove stopwords or not when performing tokenization
* `expand_btn_to_text`: to consider TextView or not when looking for a match for a Button/ImageButton
* `cross_check`: it is deprecated and can be ignored
* `reset_data`: to re-initialize/re-install the app or not when we need to re-launch the app during transfer.

Generally, a setting of `use_stopwords=True, expand_btn_to_text=False, reset_data=True` should work for most apps.

## What are the instructions to run CraftDroid on a new pair of subjects in general?

1. Prepare the test cases (json files) for the donor and recipient apps
2. Prepare the solution file by labeling the mapping between the source and target GUI events in the test cases
3. Run Apktool and Model Extractor to get static information for the source app if possible
4. Run `Explorer.py` to start the transfer
5. Evaluate the generated tests using `Evaluator.py`

# Publication

* J. Lin, R. Jabbarvand and S. Malek, "Test Transfer Across Mobile Apps Through Semantic Mapping," 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), San Diego, CA, USA, 2019, pp. 42-53, doi: 10.1109/ASE.2019.00015. ([pdf](https://www.ics.uci.edu/~seal/publications/2019_ASE.pdf))
