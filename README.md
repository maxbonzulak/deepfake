## Recreating "*Not made for each other*" Results on DFDC Dataset

*Please see the orignal author's README in the ORIGINAL_README folder.*

The "*Not made for each other*" paper proposes a method for detecting deepfake videos through audio/visual dissonance. I have been using the papers' associated github code to train and test the model on the DFDC dataset. This readme will outline my progress and provide directions to repeat the process.

In order to work with the model, some prerequisite skills will be necessary:
 1. Everything is written in python. We recommend proficiency in python or another object oriented language before working on this project.
 2. A basic familiarity with Numpy will be necessary. If you are not already familiar with NumPy, take a look at some of the basics [here](https://numpy.org/devdocs/user/quickstart.html) or [here](https://numpy.org/devdocs/user/absolute_beginners.html).  
 3. Finally, you will need to be familiar with Pytorch. If you are not already familiar with using Pytorch for deep learning, take this 60-minute tutorial: [Deep Learning With Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html#).

 With the prerequisites covered, we can move to the paper:
 [Not made for each other– Audio-Visual Dissonance-based Deepfake Detection and Localization](https://arxiv.org/pdf/2005.14405.pdf)
                                                          
 The full code for the paper is no longer available on github. However, the code is available [here](https://drive.google.com/file/d/1K_0gsamanFFgrTmmJACZs8vADYCm1jA9/view) on Prof. Bharati's google drive. You may need to request access. 

From here on out, the procedure is mostly explained in the code's original readme file (located in ORIGINAL_README). I recommend looking at that, but I will summarize and supplement the procedure for our purposes. I will reference some files that you can find in my [repository](https://github.com/maxbonzulak/deepfake/tree/main/ACM_MM_2020).

1.  Follow the steps given in  `conda_requirements.txt`.
2.  Run the command:  `pip install -r requirements.txt`.
*note: I had trouble with this command. I instead had to type the command separately for each line of requirements.txt, omitting the version in some lines.*
Now, your virtual environment is set up. Every time you want to work on the project, you can activate the environment with the command `conda activate deepfake`
3. Download DFDC dataset from  [here](https://www.kaggle.com/c/deepfake-detection-challenge/data). I started off with the small sample dataset. After setting up Kaggle, you can download the small dataset with the command `kaggle competitions download -c deepfake-detection-challenge`. The small dataset did not produce great results, so you'll want to download the larger dataset. Unfortunately, this file isn't downloadable through the kaggle API, and you need to include cookies to download. [This](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121194#latest-692892) was the best workaround I found to download the large dataset from the command prompt.
4. Store the train and test videos as follows:
```
{your_path}/ACM_MM_2020/train/real/{videoname}.mp4  
{your_path}/ACM_MM_2020/train/fake/{videoname}.mp4  
{your_path}/ACM_MM_2020/test/real/{videoname}.mp4  
{your_path}/ACM_MM_2020/test/fake/{videoname}.mp4
```
- To sort the videos into their appropriate folders, I wrote a script in the first cell of `sort_real_fake.ipynb` that consults the `metadata.json` file. You'll want to change the file locations based on where the data is coming from/going, and change the counter value based on how many videos you want to allocate for training vs testing.
5. To pre-process the videos, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test`. After this, you can feel free to delete the `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test`.
6. Collect video paths in csv files by running the command `python write_csv.py --out_dir .` .
7. We've reached the training step! Run `python train.py --epochs 50 --out_dir .` to train. In the repository, there is a modified train file called `train2.py`, along with some of its dependencies. This version chooses the appropriate MDS threshold for the current dataset, which is important in minimizing loss.
8. To test the model, run `python train.py --test log_tmp/deepfake_audio-224_r18_bs8_lr0.001/model/model_best_epoch48.pth.tar --out_dir .` The path and name of your model may be different, so look in your own path to make sure. Next, run `python test.py`. The output will be the AUC score for your test! My repository has `test2.py`, which gives some more analysis, such as a confusion matrix. 

Next steps:
Using the limited dataset yielded an AUC of about 0.58. So, the next step is to see if we can improve this with the full dataset. The researchers claim a AUC about .91 on the DFDC, so we hope the results improve on the larger set. Also, some analysis on the number of epochs may be valuable. 