# OpenMMlab实战营

## 第一次课2023.2.1

图像分类、图像检测、图像分割、较少用的Localization(把一个物体分出来)

<img src="../images/image-20230201194613943.png" alt="image-20230201194613943" style="zoom:80%;" />

像素层面：语义分割（不管像素重叠，也不区分单个物体）、实例分割（区分单个物体）

<img src="../images/image-20230201194807227.png" alt="image-20230201194807227" style="zoom:80%;" />

此外还有关键点检测

<img src="../images/image-20230201194833386.png" alt="image-20230201194833386" style="zoom:80%;" />

更形象的看一下

<img src="../images/image-20230201194910123.png" alt="image-20230201194910123" style="zoom:80%;" />

像素粒度：细粒度 / 与位置有关的精细力度

<img src="../images/image-20230201195107457.png" alt="image-20230201195107457" style="zoom:50%;" /><img src="../images/image-20230201195232957.png" alt="image-20230201195232957" style="zoom:50%;" />

深度神经网络在不同的处理任务中就对应了不同的模型（RNN/Transformer/GNN等等）

<img src="../images/image-20230201195601703.png" alt="image-20230201195601703" style="zoom:50%;" /><img src="../images/image-20230201195705131.png" alt="image-20230201195705131" style="zoom:50%;" /><img src="../images/image-20230201195725698.png" alt="image-20230201195725698" style="zoom:50%;" /><img src="../images/image-20230201195755752.png" alt="image-20230201195755752" style="zoom:50%;" />

除了视觉还要综合各种传感器等等。

在视频的理解中，我们如何剪辑出我们需要的部分，或者时间点。

<img src="../images/image-20230201195851881.png" alt="image-20230201195851881" style="zoom:50%;" /><img src="../images/image-20230201200009883.png" alt="image-20230201200009883" style="zoom:50%;" />

初期的特征是人为给定的

<img src="../images/image-20230201200506524.png" alt="image-20230201200506524" style="zoom:80%;" />

2012年之后的深度学习时代，人们就开始让机器去自己学

<img src="../images/image-20230201200554229.png" alt="image-20230201200554229" style="zoom:80%;" />

到现在：用海量数据，非常深，参数非常多的大模型不断出现。

<img src="../images/image-20230201200711874.png" alt="image-20230201200711874" style="zoom:80%;" />

我们可以做什么

<img src="../images/image-20230201200809870.png" alt="image-20230201200809870" style="zoom:80%;" />

OpenMMlab发展

<img src="../images/image-20230201201010933.png" alt="image-20230201201010933" style="zoom:50%;" /><img src="../images/image-20230201201026849.png" alt="image-20230201201026849" style="zoom:50%;" />

一些部署介绍

<img src="../images/image-20230201201241774.png" alt="image-20230201201241774" style="zoom:80%;" />

一些OpenMMlab的算法框架

<img src="../images/image-20230201201637257.png" alt="image-20230201201637257" style="zoom:50%;" /><img src="../images/image-20230201201654653.png" alt="image-20230201201654653" style="zoom:50%;" /><img src="../images/image-20230201201738024.png" alt="image-20230201201738024" style="zoom:50%;" /><img src="../images/image-20230201202437459.png" alt="image-20230201202437459" style="zoom:50%;" />

对于自己的数据集我们可以直接使用OpenMMlab中的算法来进行操作。可以看到里面的复现了很多最新论文的方法。

<font color="red">OpenMMlab是基于pytorch的，但又不完全基于原生pytorch，因为有很多新方法仅靠pytorch无法完成。</font>

<img src="../images/image-20230201202704712.png" alt="image-20230201202704712" style="zoom:50%;" /><img src="../images/image-20230201202728167.png" alt="image-20230201202728167" style="zoom:50%;" />

作为开发者我们仅需会掉包即可。

底层的内容如果不是做版本升级不是很需要学习。

<img src="../images/image-20230201202932465.png" alt="image-20230201202932465" style="zoom:80%;" />

在OpenMMlab的官网或者github中给出的其实就是该领域中的必读论文，我们也可以不去读综述。

机器学习与神经网络的简介

<img src="../images/image-20230201203636497.png" alt="image-20230201203636497" style="zoom:50%;" /><img src="../images/image-20230201204136310.png" alt="image-20230201204136310" style="zoom:50%;" />

大道至简都是从数据出发去分类去挖掘去检测

<img src="../images/image-20230201204152327.png" alt="image-20230201204152327" style="zoom:80%;" />

机器学习的基本流程：

<img src="../images/image-20230201204802638.png" alt="image-20230201204802638" style="zoom:80%;" />

通常batch取2的指数次方

<img src="../images/image-20230201205726169.png" alt="image-20230201205726169" style="zoom:80%;" />

<img src="../images/image-20230201205843363.png" alt="image-20230201205843363" style="zoom:80%;" />

神经网络+CNN介绍了一会

研究方向推荐：

<img src="../images/image-20230201211322746.png" alt="image-20230201211322746" style="zoom:80%;" />