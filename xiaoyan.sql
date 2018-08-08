-- phpMyAdmin SQL Dump
-- version 4.6.5.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: 2018-01-22 06:15:16
-- 服务器版本： 10.1.21-MariaDB
-- PHP Version: 5.6.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `xiaoyan`
--

-- --------------------------------------------------------

--
-- 表的结构 `answer`
--

CREATE TABLE `answer` (
  `answer_id` int(11) NOT NULL,
  `answer_text` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- 转存表中的数据 `answer`
--

INSERT INTO `answer` (`answer_id`, `answer_text`) VALUES
(1, 'UIC是由内地与香港两所公立大学——北京师范大学和香港浸会大学联合创办、具有独立法人资格的普通本科高等学校，实行董事会领导下的校长负责制。作为首家内地与香港高等教育界合办的大学，UIC的定位是成为“一所为中国和世界服务的新型博雅大学”。UIC以创建内地首家博雅大学，培育国际精英人才为使命，旨在通过在内地率先推行博雅教育，促进中国高等教育国际化。'),
(2, '中外合作办学。'),
(3, 'UIC校长吴清辉教授。吴清辉教授早年留学澳大利亚墨尔本大学并获化学工程学位及化学硕士学位，后在加拿大英属哥伦比亚大学（UBC）获化学博士学位。吴教授在化学领域“多相催化”方面的研究，深得海内外学术界的称许。他在香港大学及香港浸会大学从事高等教育近四十年，于2001年被委任为香港浸会大学校长，2010年离任。此外，吴教授积极参与公共事务，曾出任的公职包括香港事务顾问、香港特区筹备委员会预备工作委员会委员、香港特区筹备委员会委员、香港特别行政区立法会议员、第九、十及十一届全国人大港区代表、香港政府研究拨款委员会委员、香港科技大学校董、香港科技园董事；吴教授现任UIC校长及多间内地大学的客座教授。2005年，吴教授获香港特别行政区颁发金紫荆星章，以表扬他在公共及社会服务方面的卓越表现，尤其在推动高等教育发展方面，建树良多。'),
(4, '亚热带季风气候。'),
(5, 'UIC校园依山傍湖，环境优美。校园内所有教室配置先进的多媒体设备，专门设有实验室、计算机中心、影视广播中心、演讲厅、学习资源中心、师生活动中心、学生活动广场、高尔夫球练习场、攀岩馆、皮影戏剧艺术馆、户外体验拓展基地。学生宿舍出入口大堂采用门禁、监控系统，学生出入宿舍需刷校园卡，宿舍管理规范、安全。同时，学校致力营造有着良好归属感及多元丰富的舍堂文化，组织各舍堂宿生会，让来自海外及中国内地、港澳的同学们汇聚一堂，让同学们在舍堂环境中培养多元化的价值观及世界观，提升生命的疆界和包容力。'),
(6, 'UIC为开放式校园，以海纳百川的心态欢迎所有的考生和家长到学校参观体验。'),
(7, '学校地处珠海市高新区，附近高校林立，交通生活方便。'),
(8, 'UIC校园占地300亩，总建筑面积约26.4万平方米，其中教学区域建筑面积约19.5万平方米，是金凤路教学楼建筑面积的5倍，可容纳6000名本科生及1000名教职员。'),
(9, '学校地址：珠海市香洲区金同路2000号。邮编：519085。招生办电话：0756-3620011、3620022、3620033、3620044.'),
(10, '学校在建新校区位于唐家湾会同古村，将于2017年9月投入使用，届时我们将正式搬入位于会同的新校园'),
(11, '我校现有在校生5300余人。'),
(12, '学校官网的网址是：www.uic.edu.hk'),
(13, '学校招生网：www.uic.edu.hk/admission'),
(14, '我校招生办上班时间为周一至周五，早上8:30-12:00,下午13:00-17:30。'),
(15, '招生热线：（0756）3620011、3620022、 3620033、 3620044\r\n传真号码：（0756）3620047\r\n电子邮箱：admission@uic.edu.hk\r\n学校网址：http://www.uic.edu.hk\r\n招生信息网：http://www.uic.edu.hk/admission'),
(16, '招生热线为校内办公电话，按话费标准收取，不收取服务费。'),
(17, '我校各省均成立报考咨询QQ群。'),
(18, '我校教务处的电话是：0756-3620303。'),
(19, '我校学生处的电话是：0756-3620227。'),
(20, 'UIC以丰富多彩的大学生活，让学生在体验中学习和成长，实现学生自我管理、成长和超越，培育学生成为具有服务精神的领袖人才。UIC现有兴趣社团共64个，同学们可以充分发展自己的兴趣爱好。同时学校还会定期举办各种文化学术讲座、高桌晚宴、音乐演唱会、舞台剧、展览等。'),
(21, 'UIC2017年开放日于4月29日举行。为方便考生及家长咨询，UIC每年都会循例在年中选择一天作为开放日。当天，会把各个学部、行政部门都集中起来，为家长搭建充足的咨询和了解学校各个细节的平台，让家长花更少的时间，获得更全面的入学资讯。'),
(22, 'UIC以丰富多彩的大学生活，让学生在体验中学习和成长，实现学生自我管理、成长和超越，培育学生成为具有服务精神的领袖人才。UIC现有兴趣社团共64个，其中学术类10个，社会企业类4个，文化艺术类15个，体育运动类25个，义工服务类5个，其他类5个，同学们可以充分发展自己的兴趣爱好。'),
(23, '学校采用香港浸会大学的标准，面向海外，全球招聘。师资来自全球三十多个国家和地区，其中77%在职专任教师来自内地以外地区。截止至2016年9月，UIC教职工人数约725人，师生比例为1:7。专任教师总数242（不含助教），专任教师中博士学位比例65%，副教授及以上职称比例为33%。'),
(24, '新校园是一个绿色、开放、富于交流的教学与生活区，充分体现UIC倡导的博雅教育理念的需求。教学楼、学部综合楼、学习资源中心等公共资源围绕中央绿化带建设，形成以学生为中心，宿舍区和教学区交替排列的建设格局，使同学们的学习、生活更为便捷。文化创意中心、体育馆、大学会堂、演艺厅、行政楼、教学楼及宿舍楼等将陆续建成。'),
(25, '我校教学楼均有电梯'),
(26, '学习资源中心现有纸本藏书约23万册，种类逾20万种，中英文藏书比例约为2:3。中心订购了300多种中英文杂志和报纸；拥有与UIC各学科专业相关的英文电子图书逾17万册，中文电子图书逾33万册。另外订购有电子大英百科全书和布莱克威尔在线参考书等数百种参考工具书；数万种与UIC开设学科专业相关的英文电子学术期刊及53种多学科或单一学科数据库。读者也可通过馆员获取香港浸会大学图书馆、大陆各高校图书馆、国家图书馆、国家科技图书馆的文献。\r\n学习资源中心在每学期为读者提供每周两次的图书馆资源讲座。中心拥有丰富的英语分级读物和专业的口语培训软件，全天候播放CCTV英语新闻，同时与英语语言文化中心合作，帮助学生逐步提升英语水平。每个学年，中心都会与进出口公司合作举办两次学术图书展，展出与UIC学科相关的优质的外文原版学术图书。'),
(27, '图书馆对外开放，只要拿身份证登记即可。学习资源中心纸本藏书约23万册，种类逾20万种，中英藏书比例约为2:3，是中国外语书籍馆藏量最丰富的图书馆之一。'),
(28, '我校有室内恒温游泳馆。'),
(29, '学校有多个室内篮球馆及室外篮球场。'),
(30, 'UIC学习氛围浓厚，校风积极上进。扎实的专业基础，全面素质的发展，以及较好的英文能力，让UIC学子在各类比赛及活动中锋芒渐露。'),
(31, '近年来我校已获得的8项国家自然科学基金（NSFC）立项资助，资助金额达351万元。涵盖的学科有：数学，生物化学，信息科学和医学领域。同时还获得教育部人文社科基金、广东省自然科学基金、珠海市人文社科基金及其它产学研项目资助。'),
(32, '2016年UIC获得教育部批准开展研究生教育，招收研究型硕士及博士生，成为珠海首个全部在珠海本土培养从本科生到博士生的院校。参照国外和香港的教育模式，我校研究生课程分为研究型和授课型两种类型。'),
(33, '学校设有工商管理、人文与社会科学、理工科技和文化与创意四个学部，开设二十多个本科专业。'),
(34, '我校没有双学位课程，但学生在校期间可以副修课程。'),
(35, '为培育更多复合型的优秀人才，提高学生综合素质和升学就业竞争力，学校为本校学生提供副修课程。副修课程的设计吸收了香港与世界各地的经验，与国内高校的辅修不尽相同，在一定程度上可以说是教学上的创新。课程为学生提供了在修读本专业之外，更加系统、深入地修读其他学科的机会，增强学生在学习上多元化的自由度。学生可以根据自己的兴趣爱好、职业规划等选择跨专业的学习，拓宽知识面，成为复合型人才。注：完成课程后，同学们须在毕业前向学校申请副修资格核准，审查通过者其副修专业名称将会显示在毕业成绩(Transcript)上，而非记录在学位证书上。'),
(36, '学校设有新生入学奖学金、华信奖学金、广东省提前批第一批本科新生入学奖学金、广东省综合评价录取奖学金、珠海籍新生专项奖学金、在学奖助学金。具体请登录UIC招生信息网查看。'),
(37, 'UIC海外交流学习旨在让学生通过融入国外的学习生活环境，感受另一种文化和教育，扩大思考的角度，更深刻地理解人性和社会，为将来就业或继续深造打下更坚实的基础。UIC学生国际交流学习项目有两种，即海外暑期课程和交换生项目。针对国际学生，UIC还有国际留学生，交换生项目和海外学生短期培训项目。'),
(38, '我校招收外国留学生。'),
(39, '截止至2016年10月，男女比例，女生占总人数67%，男生占33%。'),
(40, '新生入学后，可于大一下学期，根据教务处的时间要求，提出转专业的申请意愿。学校在尊重学生意愿的基础上，结合学校的教学资源，并严格按照学校规定的程序，允许部分优秀学生申请专业调整。'),
(41, '每个学生在校期间均有一次转专业的机会。'),
(42, '转专业看的是大一期间的学习成绩，不需要花钱。'),
(43, '详情可参考学校招生信息网：www.uic.edu.hk/admission'),
(44, '原则上，理科生可以报考我校所有专业。具体请以你省教育考试院公布的招生专业目录为准。'),
(45, '文科生可以报考除统计学、应用心理学、计算机类、环境科学、食品科学与工程、金融数学以外的所有专业。'),
(46, '工商管理学部、人文与社会科学学部及文化与创意学部下属的专业都是文理兼招。'),
(47, '请考生结合自己的兴趣爱好、能力特长及将来的发展方向专业专业，我校在入学时对男女无特殊要求。'),
(48, 'GPA英语全称是Grade Point Average，意思就是平均成绩点数，UIC课程的GPA满分是4.0分。平均绩点（GPA）是体现学生学术表现的重要指标，是学生获得的所有绩点总和除以所有应获学分总和所得的结果。'),
(49, '我校毕业生大多申请境外高校的研究生，无保研体制。'),
(50, '各专业都可以申请国内及境外的研究生。'),
(51, '我校每年有超过六成的学生选择出国继续深造。'),
(52, 'UIC学生经四年在校学习，各科成绩合格，即获颁北京师范大学-香港浸会大学联合国际学院本科毕业证书（教育部电子注册）和香港浸会大学学士学位证书（内地、香港及国际范围均获认可）。'),
(53, '我校提供多种形式的兼职机会给学生，一般薪酬是18元/小时。'),
(54, '为鼓励学生毕业或出境升学后在珠海及珠三角服务，UIC每年举办职业博览会。职业博览会与招聘会不同，它包括企业展、职业讲座、招聘咨询和现场面试，让学生了解各个行业具有代表性企业的发展，通过与企业近距离的交流，更好地明确职业发展方向。'),
(55, '截至2017年2月底，UIC已培养八届约7,000名本科毕业生，遍布世界各地。UIC毕业生深受跨国企业、国企以及政府和公共组织等机构青睐，如中国银监局、四大国有银行、中央电视台、外交部、中石油、南方电网、香港商报、香港中国银行、毕马威、德勤、安永、普华永道四大会计师事务所、BP、IBM、微软等，同时还有多名校友考取香港会计师及香港律师执业资格，选择创业的学生也不在少数。'),
(56, '2017年9月，我校将整体迁入会同新校园。'),
(57, '工商管理学部：会计学、财务管理、经济学（应用经济）、电子商务（电子商务与资讯系统管理）工商管理类【人力资源管理，市场营销（市场营销管理），工商管理（创业与创新管理）】；人文与社会科学学部：新闻学、国际政治（政治与国际关系）、社会学（社会工作与社会行政）、公共关系学（公共关系与广告)、外国语言文学类、【英语，翻译】；理工科技学部：应用心理学、食品科学与工程、环境科学、金融数学、统计学、计算机类、【计算机科学与技术，数据科学与大数据技术】；文化与创意学部：文化产业管理（文化创意与管理）、传播学【电影电视/媒体艺术与设计】。'),
(58, 'UIC所有专业的课程都是按照香港浸会大学的学术标准进行设置，都达到国际标准，同时符合市场需求和国际趋势，师资配备和软硬件设施上每个专业也比较均衡。关于专业志愿选择，首先一定要根据孩子的自身兴趣爱好和未来生活及职业规划，其次通过学校教务处网页下载各专业手册（handbook），或查阅UIC招生手册，结合每个专业的培养目标及四年学习计划，最后才做出专业志愿选择，不要盲从。'),
(59, '海南考生高考英语单科成绩须达到800分或以上，同时高考文化分总分须满足：文科考生总分达到740分或以上；理科考生总分达到700分或以上。'),
(60, '浙江考生高考英语单科成绩须达到127分或以上，同时高考文化分总分须达到640分或以上。'),
(61, '江苏考生高考英语单科成绩须达到102分或以上，同时高考文化分总分须满足以下要求：文科考生总分高出一本线40分或以上；理科考生总分高出一本线50分或以上；文理科选测科目AA。'),
(62, '山东新生入学奖学金获奖条件：考生高考英语单科成绩须达到127分或以上，同时高考文化分总分须满足以下要求：文科考生总分达到590分或以上；理科考生总分达到630分或以上。'),
(63, '考生高考英语单科成绩须达到127分或以上，同时高考文化分总分须达到560分或以上。'),
(64, '考生高考英语单科成绩须达到127分或以上，同时高考文化分总分须满足以下要求：文科考生总分高出一本线60分或以上；理科考生总分高出一本线90分或以上。'),
(65, '华信奖学金为当年学费一次性减半，获奖条件及评定办法详见学校招生办公室《2017年UIC华信奖学金实施办法》。'),
(66, '广东省提前批第一批本科批次报考我校的考生；考生高考英语单科成绩须达到127分以上；同时，文史类考生高考文化课总分至少达到广东省第一批本科线60分以上，理工类考生高考文化课总分至少达到广东省第一批本科线90分以上；经学校奖助学金委员会评定，可获得广东省提前批第一批本科新生入学全额奖学金。'),
(67, '广东省提前批第一批本科批次报考我校的考生；考生高考英语单科成绩须达到127分以上；同时，文史类考生高考文化课总分至少达到广东省第一批本科线40分以上，理工类考生高考文化课总分至少达到广东省第一批本科线60分以上；经学校奖助学金委员会评定，可获得广东省提前批第一批本科新生入学半额奖学金。'),
(68, '对于综合评价批次录取的考生，综合总成绩在该批次珠海籍录取考生中文理科排名各前5名的新生，学校给予一次性第一学年学费减半的入学奖励；对于其他综合评价批次录取的珠海籍新生，学校给予一次性第一学年学费减免1万元人民币的入学奖励。\r\n对于提前批第一批本科批次录取的珠海籍新生，学校给予一次性第一学年学费减免1万元人民币的入学奖励。\r\n如学生总录取成绩优异，可获得学校广东省提前批第一批本科新生入学奖学金或广东省综合评价录取奖学金，获奖条件及审核办法详见《2017年UIC广东省综合评价录取奖学金实施办法》及《2017年UIC广东省提前批第一批本科新生入学奖学金实施办法》。\r\n珠海籍新生专项奖学金与学校2017年广东省综合评价录取奖学金及2017年广东省提前批第一批本科新生入学奖学金不可兼得。'),
(69, '符合以上申请资格的广东省考生，被我校2017年综合评价录取并正式就读的，文科考生高考总成绩至少高出广东省第一批本科线30分或以上，理科考生高考总成绩至少高出广东省第一批本科线50分或以上；且综合评价录取总成绩排名在文理各前20名的新生，经学校奖助学金委员会评定，给予当年学费减半，原则上为期四年。'),
(70, '奖学金的发放原则上为期四年，但每学年结束后须进行审核，审核成绩要求为：学生前一学年度的平均成绩须在本年级或本专业前10%以内，次学年方可续得奖学金。如遇特殊情况，学校奖助学金委员会将视情况酌情进行评定。'),
(71, '学生在校期间，学校还特别设置了以下奖学金，嘉奖学业成绩优秀及各方面均表现优异的学生：\r\n1、一等奖学金\r\n2、二等奖学金\r\n3、曾宪博教授奖学金\r\n4、冯燊均先生国情国学教育优秀学生奖学金\r\n5、许嘉璐全人教育奖学金\r\n6、广东省政府来粤留学生奖学金(针对国际生)\r\n7、联通奖学金\r\n8、古楚璧伉俪助学金\r\n9、国家奖学金\r\n10、国家励志奖学金\r\n11、国家助学金\r\n\r\n注: 欲了解上述奖助学金的申请及评选方法，请浏览网页http://uic.edu.hk/sao，点击奖学金栏目。评定准则有可能每年进行调整，最终以奖学金及经济援助委员会评定为准。'),
(72, 'UIC参照香港浸会大学的课程设置，根据内地和港澳地区的发展和对人才的需求，提出专业设置方案，经由香港浸会大学相关专业的教授专家共同商讨提出具体方案，报请香港浸会大学教务委员会审批，之后才可具体实施。详细专业设置请查看UIC招生信息网。'),
(73, '为保证教学质量，学校建立了一套完整的课程评定程序，并成立了一个由香港浸会大学校长领导的品质保证机构。该机构负责组织教学评估，对师资学术资格的鉴定，并组织包括香港浸会大学职员在内的外校评测官，定期来校检查。'),
(74, '香港学位不同于内地，只有学业水平较高者才可获得此项殊荣，学生获得哪一个等级的学位证书是根据本人在学期间的cGPA来确定。香港浸会大学学士学位的分类如下：\r\nFirst Class (甲等)\r\nSecond Class (Division Ⅰ) (乙等一级)\r\nSecond Class (Division Ⅱ) (乙等二级)\r\nThird Class (丙等)\r\nPass (颁发一般学位)'),
(75, 'UIC以博雅教育为办学理念，不仅注重专业教育更强调通识教育和全人教育。'),
(76, 'UIC博雅教育注重学生心智的开启与扩展、见识的广博与洞明，以及人格的健全和养成，而非局限于某一狭窄领域的知识和技术的传授。UIC致力于为中国内地打造第一所将专业教育与通识教育相结合，多元化、跨学科的博雅大学。UIC希望培养既能深谙中国的社会与文化，又具备全球眼光和视野的精英人才。 '),
(77, '全人教育(Whole Person Education)是UIC创新博雅教育的核心理念和灵魂，它一方面是对香港浸会大学“全人教育”理念的传承，另一方面又针对全球化时代的特点以及中国的优秀文化传统和具体国情有所创新。全人教育不仅注重知识传授和技能习得，而且要使学生在身体、智力、道德、审美、批判性思维、创造性、精神和价值操守等方面都得到发展，旨在培养博雅通达、健全发展的人。\r\n    UIC全人教育共有七个学习模块：情绪智能、体验拓展、义工服务、环境意识、体育文化、艺术体验和逆境管理。UIC全人教育建有较为完善的学习拓展基地，寒暑期还有丰富多彩的学生实践项目。'),
(78, '“四维教育”是北京师范大学-香港浸会大学联合国际学院（简称UIC）独特的创新教育理念。“四维”，即师（学校）、生（学生）、家（家庭）、国（社会）。它以学生为核心，集合学校、家庭和社会的力量，透过四方的良性互动，通过对学生在学习、生活、家庭、心理、职业发展等方面的全方位关爱与服务，结合学生的自我教育，完善以学生为本的教育模式，达到培育全人的目标并成功实现学生向社会的顺利过渡。'),
(79, 'UIC遵循先培养听说能力，再提高写作水平的教学原则，培养学生尽快适应全英文教学环境和专业学习需要。\r\n新生入学后即进行英语统一分班考试，根据学生的不同程度进行英语强化训练，由英语语言中心（ELC）提供：\r\n一对一的写作中心专业辅导，\r\n二十人每组的小班教学，\r\n三小时每周的英语专门课程，\r\n四年贯穿的小班制英语学习。 \r\n从历年学生的学习和考试情况看，大部分学生能够较好的适应全英文教学环境。'),
(80, 'UIC倡导全人教育，并贯彻在学生的宿舍生活中，目前，我们把宿舍分为八个苑舍，博雅苑、创雅苑、文雅苑、寰雅苑、卓雅苑、科雅苑、智雅苑、德雅苑。我们邀请来自学部学部的教授担任各个苑舍的舍监，参与苑舍管理；每个栋楼有一位舍堂主任（相当于辅导员），与学生一起住在宿舍，近距离地与学生接触；另外，我们也聘请了高年级的学生担任学生舍堂导师，一起参与到苑舍文化建设中来。目前，已形成传统的苑舍文化传统活动有：苑舍高桌晚宴、苑舍日论坛（大一生和毕业生各一场）、以及为了增强苑舍凝聚力的苑际达人赛。此外，每个苑舍也有举办各具苑舍特色的小型活动，让学生的苑舍生活更加多元化。'),
(81, 'UIC提供全方位的学生辅导。透过“导师关顾计划”，大一新生因专业分组, 由高年级的学生经培训成为朋辈导师作为桥梁，方便分组专业导师在四年中，在学习和生活上给予学生关怀和指导。学生在学科学习上遇到困难，亦可向相关导师请教。导师们授课后都会留在学校并预留时间对学生的学业进行辅导、与学生讨论专业上的问题。UIC为在校生就学业、情绪、交友或家庭关系等困扰提供全面而专业的心理辅导服务。我们亦开展“学习辅导班”和“一对一学习辅导”活动，由成绩优秀的同学担任朋辈导师，使得双方共同成长、共同进步。此外，学校也提供课余个人成长课程，促进学生身心灵的健康。'),
(82, '在UIC出国的途径有许多种：\r\n在UIC学习四年后，拿到香港浸会大学的学士学位证书，可以申请海外大学的研究生课程。\r\n在校期间：在UIC就读后可申请和UIC有合作关系的院校就读交换生，或参加UIC每年组织的暑期课程，包括海外实习、海外义工服务等与国外交流合作的其他活动。'),
(83, '我校不是2+2培养模式，学生四年都在珠海学习。但在校期间学生有海外学习机会，UIC学生国际交流学习项目有两种，即海外暑期课程和交换生项目。针对国际学生，UIC还有国际留学生，交换生项目和海外学生短期培训项目。'),
(84, '目前学校共推出五个副修课程，每个副修课程学生最多可修得15个学分，副修课程分别为：音乐、工商管理、财务学、公共关系与广告学以及应用心理学。'),
(85, 'UIC学生可通过通识教育分类选修课选修日语、韩语、西班牙语、德语或法语五个小语种的基础课程。'),
(86, 'UIC学生上半年期末考试在6月初结束，下半年开学注册在9月初，所以UIC暑期大概为期三个月。'),
(87, '暑期课程是UIC学子在大一至大三的暑假期间的一个短期校外留学项目，一般为期3-5周，项目根据不同国家和院校设计了很多不同专业科目和语言背景的课程。暑期课程分为香港浸会大学暑期课程和海外暑期课程。'),
(88, '2017年UIC海外暑期课程涵盖海外17所学校，主要分布在美国、加拿大、英国、法国、西班牙、新西兰、奥地利、韩国等国家，学生在不同大学学习的主题包括影视制作、明尼苏达的历史和文化、欧美文化、英语学习、韩语学习等。'),
(89, 'UIC海外暑期课程合作院校包括英国牛津大学、美国康奈尔大学、加拿大英属哥伦比亚大学、西班牙艾赛德商学院等二十多所海外知名高校。'),
(90, '暑期课程开始和结束时间通常是暑期7至8月（具体视每年情况而定）。'),
(91, '预计总费用为15000-50000人民币不等（总费用包括学费、住宿费、往返机票、保险等）。'),
(92, '部分海外暑期课程有学分，部分暑期课程可以转学分（具体查看学校相关转学分条例）。'),
(93, '暑期课程为学生自愿申请参加，2016年约有700人参加海外或香港的暑期课程。'),
(94, '除了暑期课程，学生还可选择参加UIC全人教育暑期实践活动、中国语言文化中心台湾游学营、香港浸会大学“大都会体验计划”（海外实习）等。UIC全人教育办公室与国内外各类机构（如户外体育运动协会，非政府组织、慈善公益组织、自然保护管理区、学校等）合作共同组织并开展体验全人教育模块特色的体验式学习活动，为学生提供丰富多彩的寒暑期项目。每个项目一般持续2-3个星期，传统项目有：柬埔寨，泰国等暑期义工服务项目，上海真爱梦想教练计划，中国西北部沙漠草原环境之行，台湾潜水&海洋文化游、瑞士户外体育探险夏令营等。'),
(95, '2017年秋季交换生项目中，UIC与23所海外院校合作交换，其中美国8所，英国4所，韩国3所，法国2所，日本2所，加拿大1所，德国1所，泰国1所，马来西亚1所。'),
(96, '大二、大三的在校生可以申请交换生。'),
(97, '候选人的选拔标准是IELTS 6.0或TOFEL79 和 cGPA 2.5。国际发展处将根据此标准、合作院校的名额限制、课外活动的参加情况和学生适应性来挑选合格的学生后送交院长和教务处审批。相关专业系主任或其代表会被邀请参与面试。被选中的学生可以去合作交换学习一个学期。'),
(98, '经过选拔，UIC学生可以前往与UIC签署国际交换生合约的海外合作院校，作为交换生学习一个学期。	'),
(99, '期间学生按UIC学费标准向UIC支付学费，无需支付海外院校的学费。'),
(100, '通过交换学习可以让同学们开拓国际视野，增长见识，拓宽人脉，零距离感受不同国家文化的同时，为将来升读世界名校做准备。'),
(101, '2017年秋季交换生共有61个名额，申请成功的学生将分别前往美国康克迪亚大学、加拿大圣玛丽大学、英国肯特大学、韩国首尔大学等十多所海外高校进行交换。'),
(102, 'UIC对在校生有严格的管理制度，学校《学士学位课程条例》对学生设有出勤、学术处分、毕业要求等规定。'),
(103, '对于已注册的课程，学生须按课程安排准时上课。如果学生由于自身不能控制的原因而缺勤并且希望证明情况真实以取得补偿机会（例如补交论文、作业等），须在缺勤后的五个工作日内向任课老师提交一份书面说明连同相关证明文件以供审批。\r\n对于已注册的课程，如果学生\r\na) 未经批准缺勤超过15%，或\r\nb) 出勤率低于70%（包括已获批准和未获批准的缺勤），\r\n则任课老师可对其作出适当的处罚（例如：不允许该生参加该课程的期末考试）。\r\n学生上课迟到超过十五分钟，可视为缺勤。请假必须获得教务长或副教务长明确书面批准方才有效。'),
(104, '我校有一些严格的学术规定，学术处分适用于所有本科生\r\na) 学术警告：适用于某一学期平均绩点（GPA）在1.67-1.99之间的学生；\r\nb) 留校察看：适用于某一学期的平均绩点（GPA）在1.67以下的学生；\r\nc) 勒令退学：如学生连续两学期的平均绩点（GPA）低于1.67或其他学术原因，由教务议会（Senate）作出决定。'),
(105, '平均绩点（GPA）是体现学生学术表现的重要指标，是学生获得的所有绩点总和除以所有应获学分总和所得的结果。'),
(106, '课程评分以字母等级表示，学生在一门具体课程中获得的绩点与其字母等级相对应。具体可登录UIC教务处网站了解。'),
(107, '一般情况下，学生需完成132个学分的课程体系，其中包括60学分的专业必修、选修课，24学分的自由选修课，44学分的通识教育必修课和通识教育分类选修课，以及4学分全人教育体验学习课程。'),
(108, '除了学校要求和专业要求以外，学生必须满足如下要求才能获得学士学位：\r\na) 在学校就读至少四年或者按照该专业要求被认定为全日制学生（对于获准跳级的学生，就读期限会相应减少）；\r\nb) 获得了该专业要求的所有学分，达到所有的学校要求和专业要求（对于获准跳级的学生，可以接受部分的转移学分来满足专业要求）；\r\nc) 修读的所有课程获得的累计平均绩点（cGPA）至少达到2.00，并且通过了该专业所规定的全部课程。'),
(109, '各等级的划分以累计平均绩点（cGPA）为基础：累计平均绩点3.40-4.00获得甲等荣誉学位，3.00-3.39乙等一级荣誉学位，2.50-2.99乙等二级荣誉学位，2.20-2.49丙等荣誉学位，2.00-2.19及格。'),
(110, '会计学专业认证：是指UIC会计学专业得到香港会计师公会和澳洲会计师公会的认证，从而UIC会计学专业学生可以考取香港和澳洲注册会计师专业资格。\r\n此外，学校设有专业会计转制课程，UIC非会计学专业的学生，可以通过转制课程考取香港会计师公会注册会计师专业资格。'),
(111, 'UIC的专业课程按照香港浸会大学的学术标准设置，完整地引入了香港浸会大学教学品质保障体系。\r\nUIC博雅教育课程设置分为以下几个类别：除了专业必修课、专业选修课，还有通识教育核心课、通识教育分类选修课、全人教育体验学习课程，以及自由选修课。这与我校是博雅大学的性质相吻合，注重培养全面发展、博雅通达的人才。'),
(112, '我校有开设香港浸会大学暑期课程，2016年7月，400名UIC同学修读香港浸会大学暑期课程。'),
(113, 'HKBU暑期课程面向学校全体学生。'),
(114, '(1)2013至2016级学生: 2016-2017学年第一学期的sGPA不低于1.67， 且cGPA不低于2.00；\r\n(2)2012级及以前学生: 2016-2017学年第一学期的sGPA不低于1.70，sGPA不低于2.00。'),
(115, '读一门课程的费用为人民币 13,800元 ；选读两门课程的费用为人民币 19,500元 。费用均包括学费、住宿费、入境许可证申请费、课外活动费用和交通费等。'),
(116, '学习时间： 2017年7月3日至8月2日。'),
(117, '所有课程均为3学分。对于不等同于本校课程的、或成绩低于学校成绩评估系统C的课程，学校有权拒绝批准转学分。成功转入的学分将与该生在UIC所读的课程一起计入cGPA。'),
(118, 'UIC学生申请香港的大学读研，一般同等条件下会有更多优势，例如课程接轨、语言优势等。同时，HKBU作为母校，对UIC毕业生认可度高，所以申请母校研究生相对容易。我校截至2017年2月，UIC已培养八届约7000名毕业生，据不完全统计，其中1244人进入香港的高校深造，其中714人进入香港浸会大学攻读研究生。'),
(119, '截至2017年2月底，UIC已培养八届约7,000名本科毕业生，遍布世界各地。UIC毕业生约半数前往欧美、澳洲、亚洲等多个国家及地区深造，其中超过一半毕业生进入全球前100强高校和香港八大公立高校继续深造，攻读硕士学位。如英国的牛津大学、剑桥大学、伦敦大学学院、华威大学；美国的康奈尔大学、哥伦比亚大学、纽约大学、约翰霍普金斯学院；澳大利亚的悉尼大学、墨尔本大学、新南威尔士大学；加拿大的多伦多大学；香港大学、香港中文大学、香港科技大学、香港浸会大学等。'),
(120, '学生可以结合自己的实际情况申请国外研究生，学校会举办国际研究生教育展等为学生提供良好的平台，同时教授可以为优秀的学生撰写推荐信等。');

-- --------------------------------------------------------

--
-- 表的结构 `question`
--

CREATE TABLE `question` (
  `question_id` int(11) NOT NULL,
  `question_text` text,
  `question_answer_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- 转存表中的数据 `question`
--

INSERT INTO `question` (`question_id`, `question_text`, `question_answer_id`) VALUES
(1, '什么样的大学', 1),
(2, '什么样的学校', 1),
(3, '你们学校是一所什么样的大学', 1),
(4, '怎样的大学', 1),
(5, '你们的学校是一所怎么样的大学', 1),
(6, '学校什么样', 1),
(7, '学校怎样', 1),
(8, '类别', 2),
(9, '性质', 2),
(10, '类型', 2),
(11, '类型是什么', 2),
(12, '性质是什么', 2),
(13, '类别是什么', 2),
(14, '学校是什么类别的？', 2),
(15, '学校的类别是什么？', 2),
(16, '你们学校是什么性质的？', 2),
(17, '校长', 3),
(18, '校长是谁', 3),
(19, '校长是哪位', 3),
(20, '学校的校长是哪位？', 3),
(21, '学校所在城市气候怎么样', 4),
(22, '气候', 4),
(23, '所在城市气候', 4),
(24, '天气', 4),
(25, '所在城市的天气', 4),
(26, '学校环境怎么样', 5),
(27, '环境怎么样', 5),
(28, '环境', 5),
(29, '学校环境如何？', 5),
(30, '可以来学校参观吗', 6),
(31, '参观', 6),
(32, '可以参观', 6),
(33, '可以来参观', 6),
(34, '学校周边环境怎么样', 7),
(35, '周边环境', 7),
(36, '学校的占地面积有多大', 8),
(37, '占地面积', 8),
(38, '占地面积有多大', 8),
(39, '学习有几个校区，地址在哪', 9),
(40, '几个校区，地址', 9),
(41, '校区，地址', 9),
(42, '地址', 9),
(43, '学校近期是否会搬校区', 10),
(44, '近期 搬校区', 10),
(45, '搬校区', 10),
(46, '是否搬校区', 10),
(47, '学校有多少在校生', 11),
(48, '学校有多少学生？', 11),
(49, '在校生', 11),
(50, '多少在校生', 11),
(51, '多少学生', 11),
(52, '学校官方网址是多少', 12),
(53, '官方网址', 12),
(54, '官网', 12),
(55, '学校招生网址是多少', 13),
(56, '招生网址', 13),
(57, '招生信息网', 13),
(58, '学校招生办公室的上班时间', 14),
(59, '招生办公室的上班时间', 14),
(60, '招生办公室 上班时间', 14),
(61, '招生办 上班时间', 14),
(62, '学校招生办的联系方式', 15),
(63, '招生办联系方式', 15),
(64, '招生办招生热线', 15),
(65, '招生办传真号码', 15),
(66, '招生办电子邮箱', 15),
(67, '招生热线', 15),
(68, '招生热线免费吗', 16),
(69, '热线免费', 16),
(70, '有招生咨询QQ吗', 17),
(71, '招生咨询QQ', 17),
(72, '咨询QQ', 17),
(73, 'QQ', 17),
(74, '教务处的电话是多少', 18),
(75, '教务处电话', 18),
(76, '教务处', 18),
(77, '学生处的电话是', 19),
(78, '学生处电话', 19),
(79, '学生处', 19),
(80, '学校有哪些特色的校园文化活动', 20),
(81, '特色校园文化活动', 20),
(82, '特色校园活动', 20),
(83, '特色文化活动', 20),
(84, '特色活动', 20),
(85, '活动', 20),
(86, '校园文化活动', 20),
(87, '校园活动', 20),
(88, '学校有开放日吗', 21),
(89, '开放日', 21),
(90, '学校有哪些社团', 22),
(91, '学校社团', 22),
(92, '社团', 22),
(93, '学习的师资力量如何', 23),
(94, '师资力量', 23),
(95, '师资如何', 23),
(96, '师资', 23),
(97, '学校的教学基础设施怎么样', 24),
(98, '教学设施', 24),
(99, '基础设施', 24),
(100, '教学设施怎么样', 24),
(101, '基础设施怎么样', 24),
(102, '设施', 24),
(103, '设施怎么样', 24),
(104, '学校教学楼有电梯吗', 25),
(105, '有电梯吗', 25),
(106, '电梯', 25),
(107, '教学楼电梯', 25),
(108, '图书馆多大', 26),
(109, '图书馆', 26),
(110, '图书馆大', 26),
(111, '图书馆开放', 27),
(112, '图书馆对外开放吗', 27),
(113, '图书馆对外', 27),
(114, '游泳馆', 28),
(115, '学校有游泳馆吗', 28),
(116, '篮球馆', 29),
(117, '学校有篮球馆吗', 29),
(118, '学校的校园风气怎么样', 30),
(119, '校园风气', 30),
(120, '风气', 30),
(121, '学校有哪些科研成果', 31),
(122, '科研成果', 31),
(123, '科研', 31),
(124, '有博士硕士点吗', 32),
(125, '博士点', 32),
(126, '硕士点', 32),
(127, '博士硕士点', 32),
(128, '学校设有哪些学院', 33),
(129, '设有哪些学院', 33),
(130, '设有学院', 33),
(131, '学院', 33),
(132, '学校允许修双学位吗', 34),
(133, '修双学位', 34),
(134, '允许修双学位', 34),
(135, '双学位', 34),
(136, '允许双学位', 34),
(137, '副修可以拿学位么', 35),
(138, '副修拿学位', 35),
(139, '副修学位', 35),
(140, '副修', 35),
(141, '学校奖学金有哪几类', 36),
(142, '奖学金种类', 36),
(143, '奖学金类别', 36),
(144, '奖学金类型', 36),
(145, '奖学金有哪几类', 36),
(146, '奖学金有哪几种', 36),
(147, '在校生出国留学交流的机会多吗', 37),
(148, '在校生出国机会', 37),
(149, '在校生留学机会', 37),
(150, '在校生留学交流机会', 37),
(151, '在校生出国交流机会', 37),
(152, '在校生出国留学交流机会', 37),
(153, '出国留学交流的机会多吗', 37),
(154, '出国机会', 37),
(155, '留学机会', 37),
(156, '留学交流机会', 37),
(157, '出国交流机会', 37),
(158, '出国留学交流机会', 37),
(159, '学校有外国学生吗', 38),
(160, '外国学生', 38),
(161, '学校男女比例如何', 39),
(162, '男女比例', 39),
(163, '入学后能转专业吗？需要什么样的条件', 40),
(164, '入学后转专业条件', 40),
(165, '转专业条件', 40),
(166, '贵校的学生在大学期间有几次调专业机会', 41),
(167, '几次调专业机会', 41),
(168, '几次转专业机会', 41),
(169, '转专业要花钱吗', 42),
(170, '转专业花钱', 42),
(171, '调专业花钱', 42),
(172, '转专业钱', 42),
(173, '转专业费', 42),
(174, '各专业简介', 43),
(175, '专业简介', 43),
(176, '学校理科专业都有哪些', 44),
(177, '理科专业有哪些', 44),
(178, '理科专业', 44),
(179, '学校文科专业都有哪些', 45),
(180, '文科专业有哪些', 45),
(181, '文科专业', 45),
(182, '文理兼收的专业有哪些', 46),
(183, '文理兼收的专业', 46),
(184, '文理兼收', 46),
(185, '贵校哪些专业适合女生报考', 47),
(186, '哪些专业适合女生报考', 47),
(187, '哪些专业适合女生', 47),
(188, '女生专业', 47),
(189, '适合女生专业', 47),
(190, '什么是GPA？', 48),
(191, 'GPA什么', 48),
(192, '什么GPA', 48),
(193, '学校能保研吗', 49),
(194, '能保研吗', 49),
(195, '保研', 49),
(196, '各专业都可以报考研究生吗', 50),
(197, '各专业报考研究生', 50),
(198, '报考研究生', 50),
(199, '学校的考研录取率是多少', 51),
(200, '考验录取率是多少', 51),
(201, '考验录取率', 51),
(202, '毕业后拿到的是什么文凭', 52),
(203, '毕业后拿什么文凭', 52),
(204, '毕业后拿到的文凭', 52),
(205, '毕业后文凭', 52),
(206, '学校兼职多吗', 53),
(207, '兼职多吗', 53),
(208, '兼职', 53),
(209, '校园招聘多吗', 54),
(210, '招聘多吗', 54),
(211, '招聘', 54),
(212, '应届毕业生的就业情况如何', 55),
(213, '应届毕业生就业情况', 55),
(214, '应届毕业生就业', 55),
(215, '毕业生就业', 55),
(216, '毕业生就业情况', 55),
(217, '各学院在哪个校区', 56),
(218, '学院在哪个校区', 56),
(219, '学院校区', 56),
(220, '学校开设了哪些专业', 57),
(221, '学校专业', 57),
(222, '专业', 57),
(223, '学校有哪些优势专业', 58),
(224, '优势专业', 58),
(225, '海南省新生入学奖学金获奖条件？', 59),
(226, '海南省入学奖学金条件', 59),
(227, '海南省新生入学奖学金', 59),
(228, '海南省奖学金', 59),
(229, '海南省', 59),
(230, '海南省入学奖学金', 59),
(231, '海南入学奖学金', 59),
(232, '海南奖学金', 59),
(233, '海南', 59),
(234, '浙江省新生入学奖学金获奖条件？', 60),
(235, '浙江省入学奖学金条件', 60),
(236, '浙江省新生入学奖学金', 60),
(237, '浙江省奖学金', 60),
(238, '浙江省', 60),
(239, '浙江', 60),
(240, '浙江省入学奖学金', 60),
(241, '浙江奖学金', 60),
(242, '浙江入学奖学金', 60),
(243, '江苏省新生入学奖学金获奖条件？', 61),
(244, '江苏省入学奖学金条件', 61),
(245, '江苏省新生入学奖学金', 61),
(246, '江苏省奖学金', 61),
(247, '江苏省', 61),
(248, '江苏', 61),
(249, '江苏入学奖学金', 61),
(250, '江苏省入学奖学金', 61),
(251, '江苏省奖学金', 61),
(252, '江苏奖学金', 61),
(253, '山东省新生入学奖学金获奖条件？', 62),
(254, '山东省入学奖学金条件', 62),
(255, '山东省新生入学奖学金', 62),
(256, '山东省奖学金', 62),
(257, '山东省', 62),
(258, '山东', 62),
(259, '山东省入学奖学金', 62),
(260, '山东入学奖学金', 62),
(261, '山东奖学金', 62),
(262, '上海市新生入学奖学金获奖条件？', 63),
(263, '上海市入学奖学金条件', 63),
(264, '上海市新生入学奖学金', 63),
(265, '上海市奖学金', 63),
(266, '上海市', 63),
(267, '上海入学奖学金', 63),
(268, '上海', 63),
(269, '其它750分满分的省、市、区新生入学奖学金获奖条件？', 64),
(270, '750分新生入学奖学金获奖条件', 64),
(271, '750分获奖条件', 64),
(272, '750分', 64),
(273, '750分新生入学奖学金', 64),
(274, '其它入学奖学金获奖条件？', 64),
(275, '其它入学奖学金获奖', 64),
(276, '其它入学奖学金', 64),
(277, '其它奖学金', 64),
(278, '华信奖学金获得要求是什么', 65),
(279, '华信奖学金', 65),
(280, '华信奖学金要求', 65),
(281, '华信', 65),
(282, '广东提前一批全额奖学金获得要求是什么？', 66),
(283, '广东提前一批全额奖学金要求', 66),
(284, '广东提前一批全额奖学金', 66),
(285, '广东提前全额', 66),
(286, '广东全额', 66),
(287, '全额', 66),
(288, '广东提前一批半额奖学金获得要求是什么？', 67),
(289, '广东提前一批半额奖学金要求', 67),
(290, '广东提前一批半额奖学金', 67),
(291, '广东提前半额', 67),
(292, '广东半额', 67),
(293, '半额', 67),
(294, '珠海籍新生奖学金获得要求是什么', 68),
(295, '珠海籍新生奖学金', 68),
(296, '珠海新生奖学金', 68),
(297, '珠海奖学金', 68),
(298, '珠海籍奖学金', 68),
(299, '广东综合评价录取奖学金获得要求是什么', 69),
(300, '广东综合评价录取奖学金', 69),
(301, '广东综合评价录取奖学金获得要求', 69),
(302, '广东综合评价录取奖学金要求', 69),
(303, '广东综合评价奖学金', 69),
(304, '广东综合奖学金', 69),
(305, '广东', 69),
(306, '奖学金续得有什么要求', 70),
(307, '奖学金续得', 70),
(308, '奖学金续得要求', 70),
(309, '续得要求', 70),
(310, 'UIC设有哪些在校助学金', 71),
(311, '在校助学金', 71),
(312, '助学金', 71),
(313, '奖学金', 71),
(314, '课程设置', 72),
(315, '专业课程设置', 72),
(316, '专业及课程设置', 72),
(317, '专业及课程如何设置', 72),
(318, '教学质量怎样保证', 73),
(319, '教学质量保证', 73),
(320, '教学保证', 73),
(321, '质量保证', 73),
(322, '贵校颁发的香港浸会大学的学士学位证为什么有荣誉二字？', 74),
(323, '香港浸会大学的学士学位证为什么有荣誉二字？', 74),
(324, '香港浸会大学的学士学位证为什么有荣誉', 74),
(325, '香港浸会大学的学士学位证“荣誉”', 74),
(326, '荣誉', 74),
(327, '学士学位证“荣誉”', 74),
(328, '学士学位证', 74),
(329, 'UIC的教育模式是怎样的', 75),
(330, '教育模式', 75),
(331, 'UIC的教育模式', 75),
(332, '什么是博雅教育', 76),
(333, '博雅教育', 76),
(334, '博雅', 76),
(335, '什么是全人教育', 77),
(336, '全人教育', 77),
(337, '全人', 77),
(338, '什么是四维教育', 78),
(339, '四维教育', 78),
(340, '四维', 78),
(341, '学生入读后能否适应UIC的全英文教学环境？', 79),
(342, '学生入读后适应全英文教学环境', 79),
(343, '学生入读后全英文教学环境', 79),
(344, '学生入读后全英文教学', 79),
(345, '入读后全英文教学', 79),
(346, '全英文教学', 79),
(347, '舍堂（UIC宿舍）的管理情况怎么样?', 80),
(348, '舍堂的管理情况', 80),
(349, '宿舍的管理情况', 80),
(350, '舍堂的管理', 80),
(351, '宿舍的管理', 80),
(352, '宿舍', 80),
(353, '舍堂', 80),
(354, '申请海外研究生学校会提供帮助吗？', 120),
(355, '海外研究生提供帮助', 120),
(356, '申请研究生提供帮助', 120),
(357, '申请海外研究生提供帮助', 120),
(358, '申请海外研究生', 120),
(359, '毕业生去海外读研的多吗？', 119),
(360, '毕业生海外读研', 119),
(361, '海外读研多吗', 119),
(362, '海外读研', 119),
(363, '学生申请香港浸会大学研究生有优势吗？', 118),
(364, '申请香港浸会大学研究生有优势', 118),
(365, '香港浸会大学研究生有优势', 118),
(366, '香港浸会大学暑期课程学分可转吗？', 117),
(367, '香港浸会大学暑期课程转学分', 117),
(368, '香港浸会大学转学分', 117),
(369, '香港浸会大学暑期课程学习多久？', 116),
(370, '香港浸会大学暑期课程学习多久', 116),
(371, '香港浸会大学暑期课程多久', 116),
(372, '香港浸会大学暑期课程费用大概多少', 115),
(373, '香港浸会大学暑期课程费用', 115),
(374, '香港浸会大学暑期课程报名绩点要求', 114),
(375, '香港浸会大学暑期课程绩点要求', 114),
(376, '香港浸会大学暑期课程绩点', 114),
(377, '香港浸会大学绩点', 114),
(378, '香港浸会大学要求', 114),
(379, '大几可参加HKBU暑期课程', 113),
(380, '大几参加香港浸会大学暑期课程', 113),
(381, '几年级可参加HKBU暑期课程', 113),
(382, '几年级参加香港浸会大学暑期课程', 113),
(383, '同学在校期间可以去香港浸会大学吗', 112),
(384, '在校期间去香港浸会大学', 112),
(385, 'UIC的课程结构和国内的大学有差别吗？', 111),
(386, '课程结构和国内大学差别', 111),
(387, '课程和国内大学差别', 111),
(388, '课程差别', 111),
(389, '会计学专业的专业认证是指什么？', 110),
(390, '会计专业认证', 110),
(391, '会计学的专业认证', 110),
(392, '专业认证', 110),
(393, '学位证书等级分类？', 109),
(394, '学位证等级分类', 109),
(395, '证书等级分类', 109),
(396, '学位等级分类', 109),
(397, '学位证书等级', 109),
(398, '学位证书分类', 109),
(399, '学位证书成绩要求？', 108),
(400, '学位证成绩要求', 108),
(401, '成绩要求', 108),
(402, '毕业证书学分要求？', 107),
(403, '毕业证学分要求', 107),
(404, '毕业学分要求', 107),
(405, '学分要求', 107),
(406, 'UIC评分体系？', 106),
(407, '评分体系', 106),
(408, '体系', 106),
(409, '评分', 106),
(410, 'GPA是怎么计算的？', 105),
(411, 'GPA计算', 105),
(412, 'GPA计算方式', 105),
(413, '学生会被退学吗？', 104),
(414, '会被退学', 104),
(415, '学生退学', 104),
(416, '退学', 104),
(417, '出勤率规定', 103),
(418, '出勤率', 103),
(419, 'UIC管理严格吗？', 102),
(420, '管理严格', 102),
(421, '严格', 102),
(422, '交换生的名额？', 101),
(423, '交换生名额', 101),
(424, '交换名额', 101),
(425, '为什么要申请交换生？', 100),
(426, '为什么申请交换生', 100),
(427, '为什么交换', 100),
(428, '为何', 100),
(429, '交换生的费用？', 99),
(430, '交换生费用', 99),
(431, '交换费用', 99),
(432, '交换生可以出去学习多久？', 98),
(433, '交换生学习多久', 98),
(434, '交换生出去多久', 98),
(435, '交换学习多久', 98),
(436, '交换多久', 98),
(437, '申请交换生什么条件', 97),
(438, '申请交换生条件', 97),
(439, '交换条件', 97),
(440, '申请交换条件', 97),
(441, '交换生条件', 97),
(442, '大几可申请交换生', 96),
(443, '几年级可申请交换生', 96),
(444, '大几可申请交换', 96),
(445, '几年级可申请交换', 96),
(446, '大几交换', 96),
(447, '交换生项目有哪些国家、学校', 95),
(448, '交换生有哪些国家、学校', 95),
(449, '交换生有哪些国家', 95),
(450, '交换生有哪些学校', 95),
(451, '交换国家学校', 95),
(452, '交换学校', 95),
(453, '交换国家', 95),
(454, '在UIC暑假，学生只是去修暑期课程吗？', 94),
(455, '在暑期，学生只有修暑期课程', 94),
(456, '在暑期，只是修暑期课程', 94),
(457, '只修暑期课程', 94),
(458, '暑期课程是必须参加的吗？', 93),
(459, '暑期课程必须参加', 93),
(460, '暑期必须参加', 93),
(461, '课程必须参加', 93),
(462, '必须参加', 93),
(463, '海外暑期课程有学分吗？可以转学分吗？', 92),
(464, '海外暑期课程有学分转学分', 92),
(465, '海外暑期课程有学分', 92),
(466, '海外暑期课程转学分', 92),
(467, '海外暑期课程费用多少', 91),
(468, '海外暑期课程费用', 91),
(469, '海外暑期课程学习多久', 90),
(470, '海外暑期课程多久', 90),
(471, '海外暑期课程有哪些学校', 89),
(472, '海外暑期课程学校', 89),
(473, '海外暑期课程有哪些国家', 88),
(474, '海外暑期课程国家', 88),
(475, '什么是暑期课程', 87),
(476, '暑期课程', 87),
(477, 'UIC暑期多久', 86),
(478, '暑期多久', 86),
(479, '暑期', 86),
(480, '暑期多长', 86),
(481, '暑假多久', 86),
(482, '暑假多长', 86),
(483, '暑假', 86),
(484, '在UIC可修读什么小语种', 85),
(485, '修读小语种', 85),
(486, '小语种', 85),
(487, '副修课程有哪些', 84),
(488, '副修课程', 84),
(489, '副修', 84),
(490, '学校是2+2吗', 83),
(491, '是2+2吗', 83),
(492, '2+2', 83),
(493, '出国的途径有哪些', 82),
(494, '出国途径', 82),
(495, '出国方式', 82),
(496, '留学途径', 82),
(497, '学校提供哪些辅导及个人成长帮助', 81),
(498, '提供辅导及个人成长帮助', 81),
(499, '提供辅导', 81),
(500, '提供个人成长帮助', 81),
(501, '辅导', 81),
(502, '个人成长帮助', 81);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `answer`
--
ALTER TABLE `answer`
  ADD PRIMARY KEY (`answer_id`);

--
-- Indexes for table `question`
--
ALTER TABLE `question`
  ADD PRIMARY KEY (`question_id`),
  ADD KEY `question_answer_id` (`question_answer_id`);

--
-- 在导出的表使用AUTO_INCREMENT
--

--
-- 使用表AUTO_INCREMENT `answer`
--
ALTER TABLE `answer`
  MODIFY `answer_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=121;
--
-- 使用表AUTO_INCREMENT `question`
--
ALTER TABLE `question`
  MODIFY `question_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=503;
--
-- 限制导出的表
--

--
-- 限制表 `question`
--
ALTER TABLE `question`
  ADD CONSTRAINT `question_ibfk_1` FOREIGN KEY (`question_answer_id`) REFERENCES `answer` (`answer_id`) ON DELETE CASCADE ON UPDATE CASCADE;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;