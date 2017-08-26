# 用Python建立你的第一个推荐引擎

**原文标题：Quick Guide to Build a Recommendation Engine in Python**

**作者： AARSHAY JAIN**

**翻译：张逸**

**原文链接：**[https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/](https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/)

在数据科学中，不管你是新手还是经验丰富的专业人士，自己做一些项目总是会给你的简历添彩。而我写这篇文章的目的就是让你从推荐系统开始，构建一个自己的项目。（如果你很难获得开放的数据，请在评论中告诉我）


“推荐引擎（Recommendation engines）”实际上就是一个自动化的商店柜员——你问他某个商品，他不仅会给你该商品，而且还会向你推荐可以购买的其他相关产品。可以说，他们在附加销售方面训练有素。那么接下来，就让我们来实现一个自己的推荐引擎。


而且在实际中，我们可以知道，这种推荐引擎能够很好的根据客户的历史行为推荐个性化内容，这让顾客们很高兴，同时也为网站招徕了更多的回头客。

在这篇文章中，我会介绍在Python中使用[GraphLab](https://www.analyticsvidhya.com/blog/2015/12/started-graphlab-python/)创建推荐系统的基础知识。这之后，你会对推荐系统如何工作、如何构建基本的流行度模型和协同过滤模型有一些自己的理解。

![picture1](https://www.analyticsvidhya.com/wp-content/uploads/2016/06/rec-768x1024.jpg)

这篇文章的结构如下：

 1. 推荐引擎的种类
 2. MovieLens数据集
 3. 简单的流行度模型
 4. 协同过滤模型
 5. 对推荐引擎的评估

在正式开始这篇文章之前，我要衷心感谢**华盛顿大学在Coursera上的课程[ Machine Learning Specialization ](https://www.coursera.org/specializations/machine-learning)**。它对于我理解相关概念发挥了重要作用，这篇文章也算是我学习后的一个总结。

## **1. 推荐引擎的种类**
在查看不同类型的推荐引擎之前，让我们回顾一下，看看是否可以给出一些直观的建议。考虑以下情况：
### **case1：推荐最受欢迎的物品**
一个简单的方法是推荐大多数用户都喜欢的东西。这是一个快速但是不那么好的方法，是因为这个它并没有涉及到**个性化**。

基本上，对于每个用户来说，被推荐的东西都是相同的——因为流行度定义在了整个用户群体上——所以每个人都会看到相同的结果。这听起来就好像是“一个网站建议你买微波炉，只是因为其他大多数用户喜欢它，但不在乎你是否有兴趣购买”。

令人惊讶的是，这种方法仍然被应用在类似门户网站的地方。每次你登录网站并查看bbcnews，你将看到一列“热门新闻”，它被细分为几个部分，每部分展示了阅读数最多的文章。在这种情况下，这个方法是有效的，因为：

 - 存在一些不同的分类，所以用户可以看他的感兴趣的部分。
 - 在某个时间段内，只有少数几个热点话题，所以很大概率上某个用户想要看的内容和大多数其他人的相重合
 
### **case2：用分类器来做推荐**
我们已经知道有很多的**分类算法**。下面来看看如何运用这些技术来做推荐。首先，分类器是一种参数化解决方案，因此我们只需要定义用户和项目的一些参数（特征），而对于输出结果，如果用户喜欢，输出可以为1，否则为0。在某些场景中，分类器可以起到很好的作用，因为它有如下的几个优点：

 -  体现出了个性化
 -  即使用户的历史行为记录很短或不可用，它也可以工作
 
但它还有一些主要的缺陷，使得它在实际应用中并没有太多的推广：

 - 在实际中一些特征可能是不可用的，或者即便是可用，也不能通过这些特征构建出一个好的分类器
 - 随着用户数和项目数的增长，构建一个好的分类器会变得很困难
 
### **case3：推荐算法**
  接下来我将会介绍为解决推荐问题而量身定制的特殊算法——通常有两种类型——基于内容的推荐算法和协同过滤推荐算法。你可以参考我们以前的文章来了解其具体工作原理。篇幅所限，我在这里简短回顾一下：

**1. 基于内容的推荐算法**

 - **算法思想：**如果用户喜欢某个东西，那么他也会喜欢与之相似的物品
 - 基于推荐项目本身的相似性
 - 当每个项目的上下文/属性容易确定时，该算法通常会表现的更好。比如电影推荐、猜你喜欢的歌曲等
 


**2. 协同过滤的推荐算法**

 - **算法思想：**如果用户A喜欢物品1,2,3，用户B喜欢物品2,3,4，我们认为用户A、B有相似性，所以A可能喜欢物品4，同理B可能喜欢物品1
 - 该算法完全基于用户历史行为。这个特点使得它成为了最常用的推荐算法之一，因为它不依赖于其他任何附加信息
 - 该算法有很多应用，比如大型电商平台亚马逊的商品推荐、美国运通等银行的商业建议等
 - 有以下几种类型的协同过滤算法：
     - **基于用户的协同过滤推荐（User-User Collaborative filtering）：**在这里，我们根据相似度分析找出类似的客户群，然后给目标推荐与他最像的那个客户曾经选择的东西。这个算法非常有效，但是会消耗大量的时间和资源，它主要在计算每个客户信息上花费时间。因此，对于拥有大量客户数据的平台，如果没有非常强大的并行计算系统，该算法将很难实现
     - **基于项目的协同过滤推荐（Item-Item Collaborative filtering）：**该算法跟上个算法类似，但是这次我们不关心顾客，而是努力寻找项目（物品）之间的相似性。一旦我们有了项目的相似度矩阵，我们就可以根据客户的历史购买情况给他推荐相似的商品。该算法比上个算法消耗的资源少得多，因此，对新用户的分析来讲，这个算法花的时间更少，因为我们不需要计算所有用户之间的相似性。而且一般来讲，项目的数量、特征不会发生很大的变化，所以项目的相似度矩阵会很稳定
     - **其他更简单的算法（Other simpler algorithms）：**还有其他一些方法，比如[ market basket analysis](https://www.analyticsvidhya.com/blog/2014/08/visualizing-market-basket-analysis/)，但它的预测能力要比上述算法差一些。
 
### **2. MovieLens数据集**

为了进一步分析，我们将使用MovieLens数据集，它由明尼苏达大学的GroupLens项目组收集。你可以从[这里](https://grouplens.org/datasets/movielens/100k/)下载（MovieLens 100K）。 它主要包括：

 - 943位用户对1682部电影的共**100,000条评分**（1-5）
 - 每位用户**至少评价了20部电影**
 - 每个用户简单的人口统计信息（包括年龄、性别、职业、邮政编码）
 - 电影类型信息

让我们来把数据加载到Python中。**ml-100k.zip**压缩包中有很多文件供我们使用。现在先加载其中最重要的三个文件来感觉一下这些数据。对了，我建议可以先读一读数据的说明文档（README），它会给出不同文件的信息。

```python
import pandas as pd

# 用pandas将列名传给每一个CSV文件 
# 列名在README文件中

# 读取users文件：
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

# 读取ratings文件：
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

# 读取items文件：
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')
```
现在我们来看看每个文件的内容，以便更好地了解这些数据。

 - **Users**
```python
print users.shape
users.head()
```


![用户表前5行数据][1]
由上可知，我们确实有943位用户的信息，并且每位用户由五个特征来描述——ID，年龄，性别，职业和邮政编码。

 - **Ratings**
```python
print ratings.shape
ratings.head()
```
![评分表前5行数据][2]
同样的，我们证实了表中确实有不同的用户和电影组合成的10000条评分数据，我们还注意到每个评分数据都有与之相关联的时间戳信息。

 - **Items**
```python
print items.shape
items.head()
```
 ![项目表前5行数据][3]
此数据集包含1682部电影的相关属性。其中有24列给出了特定电影的风格。 最后19列表示了每种类型，若电影属于该类型则值为1，否则为0。

现在我们要将Ratings数据集划分成用于建立模型的训练集和测试集。 幸运的是，GroupLens已经提供了预分割数据，其中测试数据对于每个用户具有10个等级，也就是说总共有9430行。下面我们把数据加载进来：

```python
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape
```
```python
Output: ((90570, 4), (9430, 4))
```
由于我们将使用GraphLab，因此可以在SFrame中进行转换。
```python
import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)
```
我们收集了所有可用的数据，可以利用这些数据进行训练和测试。 请注意，我们的数据不仅包括用户行为，还包括用户和电影自身的属性，所以基于内容的推荐算法和协同过滤的推荐算法我们都可以应用。

### **3. 简单的流行度模型**
让我们从建立基于流行度的模型开始——即根据**所有用户群体的喜好**，给每个用户推荐同样的东西。为此，我们将使用GraphLab的popularity_recommender。 可以像这样来创建一个推荐器：
```python
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
```

参数说明：

 - **train_data:** 存放所需数据的SFrame对象
 - **user_id:** 表示用户ID的列名
 - **item_id:** 表示要推荐的每个项目的列名
 - **target:** 表示用户给出的分数/评分的列名

让我们利用这个模型对用户表前五位用户作出top5的推荐，看看它是怎么工作的：
```python
#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)
```
![][4]
你注意到了吗？ 对所有用户给出的推荐是相同的——1500,1201,1189,1122,814，连顺序都一样。这可以通过在我们的ratings_base数据集中检查最高平均推荐的电影来验证：
```python
ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
```
![最高平均推荐的电影][5]
从图上可以看出，所有被推荐的电影平均评分确实是5，也就是说观看过这些电影的所有人打分都是最高分。同时也说明我们的推荐系统是按照预期工作的。但这种方式一定好吗？我们将在后续进行详细分析。

### **4. 协同过滤模型**
让我们从理解协同过滤算法的基础开始。算法核心有两个步骤：

 1. 通过相似性度量来查找类似的项目
 2. 向用户推荐与他所喜好的项目最为类似的项目

为了给你一个更深层次的概述，我们制作了一个**Item-Item矩阵**，在这个矩阵中记录了被评为相似的项目对。对于当前而言，项目（Item）就指电影。一旦我们有了上述的矩阵，我们就能根据用户对不同电影的历史评价信息给出最佳建议。需要注意的是，在算法具体实施过程中还需要注意一些东西，这将需要更深入的数学分析（原文是 mathematical introspection）。我现在会跳过这一部分。

下面介绍一下graphlab支持的3种相似性指标。

1. **Jaccard相似度（Jaccard Similarity）:**
- Jaccard相似度等于对项目A和B都进行评分的用户数除以仅对A或B进行评分的用户数
- 通常用于我们没有数值评级，只有布尔值类型的数据。比如某个商品是否被购买或者添加按钮是否被点击
2. **余弦相似度（Cosine Similarity）：**
- 余弦相似度是代表项目A、B 两向量之间夹角的余弦值
- 两向量越相近，角度越小，余弦值越大
3. **皮尔逊相似度（Pearson Similarity）：**
- 相似性用两向量之间的皮尔逊系数度量

现在让我们创建一个基于项目相似度的模型如下所示：
```python
#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)
```
![基于项目相似度推荐模型的结果][6]

在这里我们可以看到，给每个用户提供的建议是不同的。所以，这些推荐是**个性化**的。但为了具体说明这个模型的优点，我们需要一些评估推荐引擎的方法。在下一节我会重点讲述。

### **5. 对推荐引擎的评估**
为了对推荐引擎作出评估，我们可以使用precision-recall的概念。你一定很熟悉它，因为和其在分类（classification）中的思想是非常类似的。下面我来根据推荐系统定义它们：

 - **召回率（Recall）：**
  - 用户喜欢的项目里边被推荐的比例
  - 比如用户喜欢的有5个，推荐了其中的3个，那么召回率为0.6
 - **精确率（Precision）：**
  - 在所有被推荐的项目中，有多少是用户真正喜欢的
  - 比如有5个项目被推荐给用户，这之中他真正喜欢的有4个，那么精确率是0.8
 
现在让我们考虑召回率，怎么样使它最大化呢？考虑这种情况：给用户推荐所有的项目，那么用户喜欢的那部分肯定会全部被覆盖，这样我们得到的召回率将是100%！但同时考虑精确率：假如推荐了1000个商品，其中只有10个是用户感兴趣的，此时精确率仅仅为0.1%。这肯定是不行的，所以我们的目标是最大限度地同时提高精确率和召回率。

一个想法是推荐系统只推荐用户感兴趣的内容，这样，precision=recall=1，但这是一种理想化的情况，在实际应用中，我们只能是尽可能的接近它。

让我们基于上面的precision-recall概念来比较刚才建立的两个模型：
```python
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])
```
![基于precision-recall概念对两个模型进行比较][7]

这里我们可以很快观察出两个结论：

 - 基于项目的协同过滤模型很明显优于基于流行度的模型（至少十倍）
 - 在绝对水平上，即便是基于项目的协同过滤模型也表现不佳，离真正有用的推荐系统还差得很远
 
也就是说我们的推荐系统还有很大的提升空间，但在这里我就不赘述了，这个问题留给大家去思考解决。给出如下几点提示：

-  尝试利用我们拥有的一些附加上下文信息
-  考虑一些更复杂的算法，如矩阵分解等
 
最后我想说的是，和GraphLab一起，你还可以使用一些其他的开源Python软件包如下：

 - [Crab](http://muricoca.github.io/crab/)
 - [Surprise](https://github.com/NicolasHug/Surprise)
 - [Python Recsys](https://github.com/ocelma/python-recsys)
 - [MRec](https://github.com/Mendeley/mrec)
 
## 结语
在这篇文章中，我们完成了在Python中使用GrpahLab构建基本推荐引擎的全过程。首先我们了解了推荐系统的基本原理，然后加载MovieLens 100K数据集用于实验。随后，我们建立了第一个模型——一个简单的基于流行度的推荐模型——它将最受欢迎的电影推荐给了每一个用户。但由于缺乏个性化，我们又建立了基于项目的协同过滤模型，并观察了个性化在其中的体现。最后，我们讨论了precision-recall作为评估推荐引擎的指标，并通过对建立的两个模型的评估，发现协同过滤模型比流行度模型要好十倍以上。

你喜欢这篇文章吗？请在评论中给我意见/建议。


  [1]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/1.-users.png
  [2]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/2.-ratings.png
  [3]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/3.-items-1024x460.png
  [4]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/4.-popularity-recomm.png
  [5]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/5.-mean-ratings.png
  [6]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/6.-similarity-model-1-768x1002.png
  [7]: https://www.analyticsvidhya.com/wp-content/uploads/2016/06/7.-evaluate-730x1024.png
