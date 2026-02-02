#用户留言
记住，你可以随便用token，不需要给我考虑省token，我想要你用更多的token，帮我把任务做得准确无误，所以我希望你每一步做完都能检查一遍。你有需要跟我回报的内容更新在本文档里面的claude留言下面，最好加上一个## 二级标题，注明时间，每次最新的回答放在最上面，你也要参考每次的留言，方便你工作的连续。

读取我的这个文件的内容，先进行工作分析，了解我的需求，把任务分成对应的小任务，按照你的理解开始工作。
首先将现有的代码提交至github中：
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/0210meet/meg_2.git
git push -u origin main
第二，请你仔细查看01_Batch_MEG_Coregistration.py、02_MEG_Source_Estimate.py、03_Batch_ROI_MEG_sipploe.py、run_thal_cortex_spike_ccg_strong.py文件，这些代码文件是为了通过对meg进行源定位，得到皮层上癫痫灶、丘脑信号，后进行放电检测，最后去分析皮层放电是否会传到丘脑，请你认真思考，是否可行，是否正确
第三，可以查看"/data/shared_home/tlm/data/Literature/Attal and Schwartz - 2013 - Assessment of Subcortical Source Localization Using Deep Brain Activity Imaging Model with Minimum N.pdf"与"/data/shared_home/tlm/data/Literature/Wodeyar et al. - 2024 - Thalamic epileptic spikes disrupt sleep spindles in patients with epileptic encephalopathy.pdf"文献，学习他们中的方法，进一步完善第二步，在第二步中是否使用了文献中的DBA方法等等
第四：目前freesurfer安装位置为/data/shared_home/tlm/tool/freesurfer，mri数据存放在/data/shared_home/tlm/data/MEG-C/freesurfer/，meg数据存放在/data/shared_home/tlm/data/MEG-C/spikes6/，请使用/home/tuluman/.conda/envs/tlm/bin/python作为python解释器，在tlm这个虚拟环境下进行，运行结果请保存至/data/shared_home/tlm/output/，每一步的运行结果文件夹请标01_XXX,02_XXX
请仔细验证处理过程是否有误，确定没有问题，审核代码没有bug，不要重复问问题，该文件提示了的内容，完成无误后再返回

# claude 留言