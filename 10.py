# coding=utf-8
import os
from ckiptagger import WS
from ckiptagger import construct_dictionary
import pickle
import re
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def open_pkl(pkl):
    with open(f"{pkl}", 'rb') as fp:
        sorted = pickle.load(fp)
    fp.close()
    return sorted
def remove_punctuation(line):
    rule = re.compile("[^a-zA-Z0-9\\u4e00-\\u9fa5]")
    line = rule.sub('',line)
    return line
# name = "中國公共圖書館兒童閱讀推廣研究——以金陵圖書館為例"
# keyword = "公共圖書館、中國、兒童、兒童閱讀、閱讀推廣"
# abstract = "與其他地區相較，公共圖書館的兒童服務在中國起步較晚。中國的兒童閱讀推廣在近十幾年才開始起步發展。然而及早養成閱讀的習慣對於兒童非常重要。本研究旨在探討中國公共圖書館兒童閱讀推廣的現況、發展與挑戰，文獻探討之重點在於論述公共圖書館與兒童閱讀推廣之重要性、國外之兒童閲讀推廣計劃、以及中國學者之相關研究。本研究以深入訪談的方式，選取了中國江蘇省南京市金陵圖書館為案例，采訪了15位曾接觸過或正在進行兒童閲讀推廣服務工作的館員，結合本研究之研究結果歸納成幾點結論，包括：一、金陵圖書館在推行兒童閲讀推廣的過程中，需要充分考量人員的調配、活動場次、時間安排，並根據具體情況指定活動的主題和内容。二、金陵圖書館過去針對兒童進行推廣閲讀的作法涉及活動管理的各個方面。三、金陵圖書館需要應對兒童閲讀推廣過程中所遇到的各類問題。四、金陵圖書館在推廣兒童閱讀方面尋求與其他機構的合作。本研究根據研究結果提出幾點建議，包括：政府的主要支援、館員素質；提高親子閱讀的重要性；積極與兒童閲讀或教育相關的團體合作；合理規劃公共圖書館内的兒童閲讀空間；加强兒童閲讀與多媒體之融合。"
# content = "摘 要 i 目 次 iii 表 次 v 圖 次 vi 第一章 緒論 1 第一節 研究背景與動機 1 第二節 研究目的與問題 3 第三節 研究範圍與限制 4 第四節 名詞解釋 4 第二章 文獻回顧與探討 6 第一節 公共圖書館的歷史發展 6 第二節 公共圖書館與兒童閱讀推廣 10 第三節 國外兒童閱讀推廣計畫 17 第四節 中國公共圖書館的兒童閱讀推廣之發展與挑戰 22 第五節 中國公共圖書館兒童閱讀推廣相關研究 26 第三章 研究方法 31 第一節 研究進行方式 31 第二節 研究場域與對象 34 第三節 研究工具 36 第四節 資料蒐集與分析 42 第四章 研究結果 47 第一節、受訪者背景資料分析 47 第二節、金陵圖書館制定兒童閱讀推廣計劃考量之層面 49 第三節、圖書館讀者參與活動之情形 57 第四節、推廣遭遇問題與因應 60 第五節、未來兒童閱讀推廣之建議 67 第五章 結論與建議 75 第一節 研究結論 75 第二節 公共圖書館實施兒童閲讀推廣建議 79 第三節 對未來研究方向之建議 85 參考文獻 87 附錄A 半結構性訪談大綱 97 附錄B 受訪者同意書 98"
# reference = "中文文獻 于偉豔（2004）。中國圖書館的發展歷程簡述。黑龍江民族叢刊，2004(5)， 124-126。 王肖霞（2017）。公共圖書館開展少兒閱讀推廣的策略。文藝生活，2017(6)，232-234。 王梅玲（2011）。中華民國圖書館發展史。載於漢寶德、呂芳上（主編），中華民國發展史‧教育與文化下冊（560-594）。臺北市：聯經。 王國馨（2002）。小大讀書會發展歷程之研究（碩士論文）。國立臺灣師範大學家政教育研究所，台北市。取自：https://hdl.handle.net/11296/z7w82j。 王淑儀（2007）。公共圖書館的青少年閱讀推廣服務探討。臺灣圖書館管理季刊，3(2)，53-60。 王薇（2013）。日本兒童閱讀狀況和推廣活動考察。圖書館雜誌, 32(3)，70-74。 天下雜誌教育基金會（2008）。閱讀，動起來–借鏡國際成功經驗，看見孩子微笑閱讀。臺北市：天下雜誌。 中華民國（2015）。圖書館法。總統府公報，7179，58-61。取自：http://nclfile.ncl.edu.tw/files/201611/1707bb0a-478a-467a-8ff7-45a13ec6a2c0.pdf。 中國大百科全書出版社編輯部編著（1993）。中國大百科全書（1-60冊）。臺北市：錦繡。 毛慶禎（譯）（2003）。公共圖書館服務綱領：國際圖書館協會聯盟/聯合國教科文組織發展指南。臺北：中國圖書館學會。 白鳳華（2017）。公共圖書館少兒閱讀推廣及其核心思路構架。中文資訊，2017(6)，32，235。 江慧（2017）。分級閱讀視角下合肥市少年兒童圖書館閱讀推廣調查與分析（碩士論文）。安徽大學，合肥市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=Y3213969。 汤诗瑶、黄维（2017）。2017全國少兒閱讀峰會在寧波舉辦。取自：http://book.people.com.cn/n1/2017/1013/c68880-29586548.html。 李玉梅（譯）（2009）。閱讀的力量（原作者：Krashen S. D., 2004）。臺北市：心理。 李怡梅（2010）。我館少兒閱讀推廣活動的探討與思考。圖書與情報，2010(4)，104-107。 呂嘉紋（2006）。兒童閱讀方案之規劃與實施～一位國小教務主任的省思與實踐（碩士論文）。國立花蓮教育大學國民教育研究所，花蓮縣。取自：https://hdl.handle.net/11296/w937bz。 吳芝儀（譯）（1995）。質的評鑑與研究（原作者：Patton M. Q., 1990）。臺北縣：桂冠。 邱子恒（2001）。從臺大圖資系學生參加外所甄試入學考試之心路歷程思考圖書資訊學系大學部教育之發展。中國圖書館學會會報，67，109-119。 汪霖（2015）。公共圖書館的兒童閱讀推廣服務研究（碩士論文）。黑龍江大學，黑龍江市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=Y2771270。 宋建成（2012）。公共圖書館史。圖書館學與資訊科學大辭典。取自：http://terms.naer.edu.tw/detail/1679322/。 宋雪芳（2011）。省視公共圖書館當代兒童閱讀推廣與實踐。臺北市立圖書館館訊，29(1)，1-10。 苑紅梅（2017）。公共圖書館少兒閱讀推廣服務模式研究（碩士論文）。吉林大學，長春市。取自：https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201801&filename=1017158767.nh&v=MDUzODU3cWZaT1JyRkN2blZiL01WRjI2R2JLOUZ0YktxSkViUElSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnJDVVI=。 林秀滿、董秋蓉（2015）。公共圖書館兒童閱讀推廣策略初探及經驗分享。公共圖書館，1，1-17。取自： http://www.nlpi.edu.tw/Public/Publish/publib/6公共圖書館兒童閱讀推廣策略初探及經驗分享.pdf。 林松柏（2015）。公共圖書館未成年人閱讀推廣研究——以長春市為例（碩士論文）。東北師範大學，長春市。取自：https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201502&filename=1015418006.nh&v=MTMwNTZyV00xRnJDVVI3cWZaT1JyRkN2aFViL05WRjI2RzdlNUZ0SE1xWkViUElSOGVYMUx1eFlTN0RoMVQzcVQ=。 林宜和（2005）。日本的兒童閱讀日。全國新書資訊月刊，76，37-39。 林姿君（2006）。嬰幼兒家長實施親子共讀之調查研究——以臺中市為例（碩士論文）。朝陽科技大學幼兒保育系研究所，臺中市。取自：https://hdl.handle.net/11296/a6sjqx。 林美君（2009）。公共圖書館辦理推廣活動影響因素之研究（碩士論文）。輔仁大學圖書資訊學系，新北市。 取自：https://hdl.handle.net/11296/p56tbj。 周田田（2013）。面向兒童的民間公益閱讀推廣組織研究（碩士論文）。河北大學，保定市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=D350083。 周倩如（2008）。有趣的閱讀生活從公共圖書館開始－閱讀活動的規劃與推廣策略。取自：https://www.nlpi.edu.tw/FileDownLoad/LibraryPublication/20111108135757241127.pdf。 郝文靜（2017）。公共圖書館視角下我國親子讀書會發展策略研究（碩士論文）。河北大學，保定市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=D01286855。 胡幼慧（主編）（2008）。質性研究：理論、方法及本土女性研究實例．二版。臺北市：巨流。 胡述兆（1995）。圖書館學與資訊科學大辭典。臺北：漢美。 洪世昌（2011）。公共圖書館的閱讀推廣與閱讀教育資源。教師天地，172，21-27。 夏白鴿（1998）。中國歷史上第一次與圖書館事業有關的出洋考察。大學圖書館學報，16(1)，41-45。 徐冬梅（2009）。大陸兒童閱讀推廣發展報告（2000-2008）。中國圖書商報，2009。 翁興利（2004）。政策規劃與行銷。臺北：華泰文化事業。 陸曉紅（2013）。我國兒童閱讀推廣研究綜述。圖書館工作與研究，2013(9)，112-116。 陳永昌（2006）。給孩子一生最好的禮物——英國「Bookstart閱讀起步走」運動。全國新書資訊月刊，2006(4)，33-37。 陳昭珍（2003）。公共圖書館與閱讀活動。臺北市立圖書館館訊，20(4)，55-67。 陳姣（2016）。圖書館兒童閱讀推廣服務模式研究（碩士論文）。湘潭大學，湘潭市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=D01057203。 陳精芬（2006）。歐洲國家兒童閱讀活動之探討：以芬蘭、愛爾蘭、英國、瑞典及奧地利為例（碩士論文）。淡江大學資訊與圖書館學系碩士班，新北市。取自：https://hdl.handle.net/11296/j5mfvz。 陳麗君（2012）。公共圖書館推動嬰幼兒閱讀服務之研究（碩士論文）。輔仁大學圖書資訊學系，新北市。取自：https://hdl.handle.net/11296/nswj83。 教育部（2000）。積極推動全國兒童閱讀運動實施計畫。教育部公報，308，29-30。 許桂菊（2015）。英國、美國、新加坡兒童和青少年閱讀推廣活動及案例分析和啟示。圖書館雜誌, 34(4)，94-102。 張巧（2017）。合肥市蕪湖市銅陵市公共圖書館兒童閱讀推廣活動調查報告（碩士論文）。安徽大學，合肥市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=Y3213961。 張雯（2016）。國公共圖書館開展親子共讀活動的現狀及策略研究（碩士論文）。雲南大學，昆明市。取自：https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201701&filename=1016233905.nh&v=MjI4MTBqTXFwRWJQSVI4ZVgxTHV4WVM3RGgxVDNxVHJXTTFGckNVUjdxZlpPUnJGQ3JtVUwvQlZGMjZHTEc3SGQ=。 張慧麗（2012）。公共圖書館兒童早期閱讀服務基本理論問題探討。圖書館， 2012(6)，87-89。 張濤（2017）。淺談青少年兒童閱讀推廣的意義和措施。中文資訊，2017(6)，79。 黃國正（2007）。公共圖書館事業與利用。臺北市：秀威資訊科技。 董雲風（2017）。淺析公共圖書館如何促進兒童閱讀推廣。中文資訊，2017(3)，18。 馮秋萍（1998）。臺灣地區國小五、六年級兒童課外閱讀行為：以國立政治大學附設實驗學校為例（碩士論文）。淡江大學教育資料科學學系，新北市。取自：https://hdl.handle.net/11296/66kse5。 馮雲、楊玉麟（2010）。近十年來中國公共圖書館事業發展的特徵及存在問題。圖書與情報，2010(2)，41-44。 曾淑賢（2003）。公共圖書館在終身學習社會中的經營策略與服務效能。臺北市：孫運璿基金會。 曾媛（2015）。公共圖書館兒童閱讀推廣模式研究。山東圖書館學刊，2015(4)，63-66。 楊美華（2005）。繞著地球跑：世界各國的閱讀活動。載於賴甫昌、林麗娜、周孟香、林秀滿、劉俊（編），公共圖書館行銷經營（180-195頁）。臺中市：國立臺中圖書館。 楊清媚（2014）。公共圖書館推廣「閱讀起步走」政策行銷及其成效——以臺中市圖書資訊中心為例（碩士論文）。南華大學國際暨大陸事務學系公共政策研究碩士班，嘉義縣。 取自：https://hdl.handle.net/11296/2szpt4。 楊衛東（2010）。少兒圖書館在兒童閱讀推廣中的使命與擔當。新世紀圖書館，2010(4)，92-94。 詹福瑞（2008）。“共同架起兒童與圖書的橋樑——紀念國際兒童圖書節40周年暨中國兒童閱讀日系列活動”啟動儀式在首都圖書館舉行。中國圖書館年鑒，2008，154-155。 新華網（2017）。金陵圖書館「七彩夏日」少兒暑期夏令營落幕。取自：http://big5.xinhuanet.com/gate/big5/www.js.xinhuanet.com/2017-08/27/c_1121549811.htm。 臺灣中華書局股份有限公司、美國大英百科全書公司編譯（1991）。簡明大英百科全書（Concise Encyclopedia Britannica）。臺北市：臺灣中華。 齊若蘭（2003）。圖書館與閱讀運動研討會論文集。臺北市國家圖書館。 鄭辰（2013）。福州市少兒圖書館的兒童閱讀推廣活動研究——家長的認知與態度（碩士論文）。安徽大學，合肥市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=Y2321744。 鞏媛媛（2016）。山西省圖書館學齡前兒童閱讀推廣活動調查研究（碩士論文）。河北大學，保定市。取自：http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=D01080761。 劉安訓（2003）。臺北縣國小推展兒童閱讀之研究（碩士論文）。國立臺東大學兒童文學研究所，臺東縣。取自：https://hdl.handle.net/11296/bvaf2g。 劉瑩（1992）。親職教育從親子共讀起步——共讀《雪人》圖書故事書。幼兒教育年刊，5，93-108。 賴苑玲（2002）。當前國小推廣閱讀活動及師生反應之研究。臺中師院學報，16，285-308。 蟲雅各（2000）。兒童閱讀的同村協力。誠品好讀，2000(8)，36-37。 簡碧瑱、塗妙如（2012）。參與親子共讀課程家長的閱讀信念、共讀行為及幼兒語言能力之初探。人類發展與家庭學報，(14)，125-153。 英文文獻 Bracken, S. S., & Fischel, J. E. (2008). Family reading behavior and early literacy skills in preschool children from low-income backgrounds. Early Education and Development, 19(1), 45-67. Evans, M. A., & Shaw, D. (2008). Home grown for reading: Parental contributions to young children's emergent literacy and word recognition. Canadian Psychology, 49(2), 89-95. Gettys, C. M., & Fowler, F. (1996). The relation of academic and recreational reading attitudes school wide: A beginning study. Retrieved from ERIC database. (ED402568). International Federation of Library Associations and Institutions. (2015). Guidelines for children's library services. Retrieved from: http://www.ifla.org/files/libraries-for-children-and-ya/publications/guidelines-for-childrens-libraries-services-zh.pdf. International Federation of Library Associations and Institutions. (2016). IFLA/UNESCO public library manifesto 1994. Retrieved from: https://www.ifla.org/publications/ifla-unesco-public-library-manifesto-1994. Kotlor, P. & Levy, S. J. (1969). Broadening the Concept of Marketing. Journal of Marketing, 33, 10-15. Krashen, S. D. (2004). The power of reading: Insights from the research. Westport, Conn. : Libraries Unlimited. Li, Y., Morgan, L., & Li, J. (2016). Calling for children friendly community life: Voices of children and parents from china. Community Engagement Program Implementation and Teacher Preparation for 21st Century Education, 209-237. Lynch, J., Anderson, J., Anderson, A., & Shapiro, J. (2006). Parents' beliefs about young children's literacy development and parents' literacy behaviors. Reading Psychology, 27, 1-20. Lynch, J., Anderson, J., Anderson, A., & Shapiro, J. (2008). Parents and preschool children interacting with storybooks: Children's early literacy achievement. Reading Horizons, 48(4), 227-242. Moore, M., & Wade, B. (2003). Bookstart: a qualitative evaluation. Educational Review, 55(1), 3-13. Pennsylvania Library Association. (2001). The role of public libraries in children's literacy development: an evaluation report. Retrieved from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.611.4915&rep=rep1&type=pdf. W. J. Grabski Public Library in Ursus. (2001). Description of successful reading programme. Retrieved from: http://www.lifelongreaders.org/lire1/resources/case_studies_english/Reading_Finland.pdf. Walter, Virginia A. (2009). The children we serve: Five notions of childhood suggest ways to think about the services we provid. American Libraries, 40(10), 52-55."
name = "應用二次剩餘於無線射頻辨識所有權轉移與授權機制之研究"
keyword = "群體所有權授權、同態加密、所有權轉移、二次剩餘、無線射頻辨識、時效性"
abstract = "無線射頻辨識技術 (Radio Frequency Identification, RFID)因具有可一次性讀取多筆資料、標籤型態多樣、資料可重複讀寫以及具備資料讀取過程加密等特性，在物流、交通、醫療等領域被廣泛使用。隨著其應用範圍的提升，資料隱私與安全性逐漸受到重視。諸多RFID認證之機制被廣泛提出。但認證機制不足以因應RFID廣泛的運用。實際應用上，標籤的所有權（亦即標籤讀取權限）需要被轉移或授權，因而延伸出所有權轉移、群體所有權轉移以及具時效性之群體所有權授權等研究議題。然而，確保所有權轉移或授權時資料的隱私與安全性，為本論文探討之議題。 本論文中，我們分析近年來與RFID所有權相關之研究並提出新的機制，提出之機制可概分為三大部分：（一）我們改良Cao等人提出之所有權轉移協定所存在的明文計算錯誤、系統不同步及標籤被偽造等安全性威脅，並提出一更為安全與效率之所有權轉移機制，（二）我們運用同態加密達成群體聚合，並利用具輕量級計算效能的二次剩餘理論提出一群體所有權轉移機制。該機制達到雙向認證、前向/後向安全等安全性需求與縮減群體轉移時所需之計算負擔。（三）我們在更進一步的提出了具時效性之群體所有權授權，該機制支援群體所有權授權、授權提早撤銷及第e階段的授權驗證等功能，並達成上述的安全性需求及縮減群體授權所需之群體授權所需之計算負擔。上述三個部分我們都透過安全性分析、效率分析及BAN邏輯分析證明本論文提出之機制的安全性、效率及正確性且提出之機制更能夠廣泛運用於實際環境之中。"
content = "誌謝 (ACKNOWLEDGEMENTS) i 中文摘要 ii ABSTRACT iii TABLE OF CONTENTS v LIST OF TABLES vii LIST OF FIGURES viii Chapter 1 Introduction 1 1.1 Research Motivation 1 1.2 Research Subject 2 1.3 Thesis Organization 3 Chapter 2 Preliminaries 5 2.1 Quadratic Residues 5 2.2 Homomorphic Encryption 7 Chapter 3 An Improved RFID Ownership Transfer Protocol using Quadratic Residues for Cloud Applications 9 3.1 Introduction 10 3.2 Related Works 13 3.3 Review of Cao et al.’s Protocol 16 3.4 Cryptanalysis of Cao et al.’s Protocol 25 3.5 The Proposed Ownership Transfer Protocol 27 3.6 Correctness Analysis of The Proposed Ownership Transfer Protocol 33 3.7 Security Analysis of The Proposed Ownership Transfer Protocol 38 3.8 Performance Analysis of The Proposed Ownership Transfer Protocol 43 Chapter 4 A Novel Group Ownership Transfer Protocol for RFID Systems 45 4.1 Introduction 46 4.2 Related Works 51 4.3 The Proposed Protocol 54 4.4 Correctness Analysis of the Proposed Protocol 66 4.5 Security Analysis of the Proposed Protocol 70 4.6 Performance Analysis of The Proposed Protocol 77 Chapter 5 A Novel Group Ownership Delegate Protocol for RFID Systems 79 5.1 Introduction 80 5.2 The Proposed Protoocl 84 5.3 Correctness Analysis of the Proposed Protocol 97 5.4 Security Analysis of the Proposed Protocol 102 5.5 Performance Analysis of the Proposed Protocol 110 Chapter 6 Conclusions 111 References 113"
reference = "[1]Amjad Ali Alamr, Firdous Kausar, Jongsung Kim, and Changho Seo, A secure ECC-based RFID mutual authentication protocol for internet of things, The Journal of Supercomputing, pp. 1-14, 2016. DOI=http://dx.doi.org/10.1007/s11227-016-1861-1. [2]M. H. Asghar, A. Negi, and N. Mohammadzadeh, Principle application and vision in Internet of Things (IoT), in Proceedings of 2015 International Conference on Computing, Communication & Automation (ICCCA), 2015, pp. 427-431. [3]Yaser Azimi and Jamshid Bagherzadeh, Improvement of quadratic residues based scheme for authentication and privacy in mobile RFID, in Proceedings of 2015 7th Conference on Information and Knowledge Technology (IKT), Urmia, Iran, 2015, pp. 1-6. [4]Nasour Bagheri, Masoumeh Safkhani, and Hoda Jannati, Security analysis of Niu et al. authentication and ownership management protocol, IACR Cryptology ePrint Archive, pp. 1-8, 2015. [5]Michael Burrows, Martin Abadi, and Roger M Needham, A logic of authentication, in Proceedings of the Royal Society of London A: Mathematical, Physical and Engineering Sciences, 1989, pp. 233-271. [6]Shaoying Cai, Yingjiu Li, Tieyan Li, and Robert H Deng, Attacks and improvements to an RIFD mutual authentication protocol and its extensions, in Proceedings of the second ACM conference on Wireless network security, 2009, pp. 51-58. [7]Tianjie Cao, Xiuqing Chen, Robin Doss, Jingxuan Zhai, Lucas J Wise, and Qiang Zhao, RFID ownership transfer protocol based on cloud,  Computer Networks, vol. 105, pp. 47-59, 2016. [8]Chin-Ling Chen and Chih-Feng Chien,  An ownership transfer scheme using mobile RFIDs,  Wireless Personal Communications, vol. 68, no. 3, pp. 1093-1119, 2013. [9]Yalin Chen and Jue-Sam Chou,  ECC-based untraceable authentication for large-scale active-tag RFID systems Electronic Commerce Research, vol. 15, no. 1, pp. 97-120, 2015. [10]Yalin Chen, Jue-Sam Chou, and Hung-Min Sun, A novel mutual authentication scheme based on quadratic residues for RFID systems, Computer Networks, vol. 52, no. 12, pp. 2373-2380, 2008. [11]Hung-Yu Chien, De-synchronization attack on quadratic residues-based RFID ownership transfer,in Proceedings of 2015 10th Asia Joint Conference on Information Security (AsiaJCIS), Kaohsiung City, Taiwan, 2015, pp. 42-47. [12]Guo Cong, Zi-jian Zhang, Lie-huang Zhu, Yu-an Tan, and Yang Zhen, A novel secure group RFID authentication protocol, The Journal of China Universities of Posts and Telecommunications, vol. 21, no. 1, pp. 94-103, 2014. [13]Tassos Dimitriou, Key evolving RFID systems: Forward/backward privacy and ownership transfer of RFID tags, Ad Hoc Networks, vol. 37, pp. 195-208, 2016. [14]Robin Doss, Saravanan Sundaresan, and Wanlei Zhou, A practical quadratic residues based scheme for authentication and privacy in mobile RFID systems, Ad Hoc Networks, vol. 11, no. 1, pp. 383-396, 2013. [15]Robin Doss, Wanlei Zhou, and Shui Yu, Secure RFID tag ownership transfer based on quadratic residues, IEEE Transactions on Information Forensics and Security, vol. 8, no. 2, pp. 390-401, 2013. [16]Sviatoslav Edelev, Somayeh Taheri, and Dieter Hogrefe, A secure minimalist RFID authentication and an ownership transfer protocol compliant to EPC C1G2, in Proceedings of 2015 IEEE International Conference on RFID Technology and Applications (RFID-TA), 2015, pp. 126-133. [17]Kai Fan, Wei Wang, Yue Wang, Hui Li, and Yintang Yang, Cloud-based lightweight RFID healthcare privacy protection protocol, in Proceedings of 2016 IEEE Global Communications Conference (GLOBECOM), Washington DC, USA, 2016, pp. 1-6. [18]Behrouz A Forouzan and Debdeep Mukhopadhyay, Cryptography and network security (sie): McGraw-Hill Education, 2011. [19]Craig Gentry, Fully homomorphic encryption using ideal lattices, in Proceedings of the forty-first annual ACM symposium on Theory of computing, Bethesda, USA, 2009, pp. 169-178. [20]Craig Gentry, Toward basing fully homomorphic encryption on worst-case hardness, in Proceedings of Annual Cryptology Conference, Santa Barbara, USA, 2010, pp. 116-137. [21]Shafi Goldwasser and Silvio Micali,Probabilistic encryption, Journal of computer and system sciences, vol. 28, no. 2, pp. 270-299, 1984. [22]Kang Hong-yan,Analysis and design of ECC-based RFID grouping-proof protocol, Open Automation and Control Systems Journal, vol. 7, pp. 1523-1527, 2015. [23]Cheng-Ter Hsi, Yuan-Hung Lien, Jung-Hui Chiu, and Henry Ker-Chang Chang, Solving scalability problems on secure RFID grouping-proof protocol, Wireless Personal Communications, vol. 84, no. 2, pp. 1069-1088, 2015. [24]Hoda Jannati and Abolfazl Falahati, Cryptanalysis and enhancement of a secure group ownership transfer protocol for RFID tags,in Global Security, Safety and Sustainability & e-Democracy, ed: Springer, 2012, pp. 186-193. [25]Yongming Jin, Hongsong Zhu, Zhiqiang Shi, Xiang Lu, and Limin Sun, Cryptanalysis and improvement of two RFID-OT protocols based on quadratic residues, in Proceedings of 2015 IEEE International Conference on Communications (ICC), 2015, pp. 7234-7239. [26]Gaurav Kapoor and Selwyn Piramuthu, Single RFID tag ownership transfer protocols,IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 42, no. 2, pp. 164-173, 2012. [27]Gaurav Kapoor, Wei Zhou, and Selwyn Piramuthu, Multi-tag and multi-owner RFID ownership transfer in supply chains, Decision Support Systems, vol. 52, no. 1, pp. 258-270, 2011. [28]Wen-Tsai Ko, Shin-Yan Chiou, Erl-Huei Lu, and Henry Ker-Chang Chang, Modifying the ECC-based grouping-proof RFID system to increase inpatient medication safety,  Journal of medical systems, vol. 38, no. 9, pp. 66-78, 2014. [29]Cheng-Chi Lee, Chung-Lun Cheng, Yan-Ming Lai, and Chun-Ta Li,  Cryptanalysis of Dimitriou's key evolving RFID systems,  in Proceedings of the Fifth International Conference on Network, Communication and Computing, Kyoto, Japan, 2016, pp. 229-233. [30]Gaochao Li, Xiaolin Xu, and Qingshan Li,  LADP: A lightweight authentication and delegation protocol for RFID tags,  in Proceedings of 2015 Seventh International Conference on Ubiquitous and Future Networks (ICUFN), Sapporo, Japan, 2015, pp. 860-865. [31]Nan Li, Yi Mu, Willy Susilo, and Vijay Varadharajan,  Shared RFID ownership transfer protocols,  Computer Standards & Interfaces, vol. 42, pp. 95-104, 2015. [32]Zhangbing Li, Xiaoyong Zhong, Xiaochun Chen, and Jianxun Liu,  A lightweight hash-based mutual authentication protocol for RFID,  in Proceedings of Processes and Cooperation International Workshop on Management of Information, Hangzhou, China, 2016, pp. 87-98. [33]Yuan-Hung Lien, Cheng-Ter Hsi, Xuefei Leng, Jung-Hui Chiu, and Henry Ker-Chang Chang,  An RFID based multi-batch supply chain systems,  Wireless Personal Communications, vol. 63, no. 2, pp. 393-413, 2012. [34]Chae Hoon Lim and Taekyoung Kwon,  Strong and robust RFID authentication enabling perfect ownership transfer,  in Proceedings of International Conference on Information and Communications Security, Raleigh, USA, 2006, pp. 1-20. [35]Iuon-Chang Lin, Hung-Huei Hsu, and Chen-Yang Cheng,  A cloud-based authentication protocol for RFID supply chain systems,  Journal of Network and Systems Management, vol. 23, no. 4, pp. 978-997, 2015. [36]Umar Mujahid, Muhammad Najam-ul-Islam, and Shahzad Sarwar,  A new ultralightweight rfid authentication protocol for passive low cost tags: Kmap,  Wireless Personal Communications, vol. 94, no. 3 pp. 725-744, 2017. [37]J Munilla, M Burmester, and A Peinado,  Attacks on ownership transfer scheme for multi-tag multi-owner passive RFID environments,  Computer Communications, vol. 88, pp. 84-88, 2016. [38]Jorge Munilla, Fuchun Guo, and Willy Susilo,  Cryptanalaysis of an EPCC1G2 standard compliant ownership transfer scheme,  Wireless Personal Communications, vol. 72, no. 1, pp. 245-258, 2013. [39]Haifeng Niu, Eyad Taqieddin, and S Jagannathan,  EPC Gen2v2 RFID standard authentication and ownership management protocol,  IEEE Transactions on Mobile Computing, vol. 15, no. 1, pp. 137-149, 2016. [40]Kyosuke Osaka, Tsuyoshi Takagi, Kenichi Yamazaki, and Osamu Takahashi,  An efficient and secure RFID security method with ownership transfer,  in Proceedings of 2006 International Conference on Computational Intelligence and Security, 2006, pp. 1090-1095. [41]Saiyu Qi, Yuanqing Zheng, Mo Li, Li Lu, and Yunhao Liu,  Secure and private RFID-enabled third-party supply chain systems,  IEEE Transactions on Computers, vol. 65, no. 11, pp. 3413-3426, 2016. [42]Farzana Rahman, Md Zakirul Alam Bhuiyan, and Sheikh Iqbal Ahamed,  A privacy preserving framework for RFID based healthcare systems,  Future Generation Computer Systems, vol. 72, pp. 339-352, 2016. [43]Biplob R. Ray, Jemal Abawajy, Morshed Chowdhury, and Abdulhameed A. Alelaiwi,  Universal and secure object ownership transfer protocol for the Internet of Things,  Future Generation Computer Systems, 2017. DOI=http://dx.doi.org/10.1016/j.future.2017.02.020. [44]Ronald L Rivest, Len Adleman, and Michael L Dertouzos,  On data banks and privacy homomorphisms,  Foundations of secure computation, vol. 4, no. 11, pp. 169-180, 1978. [45]Kenneth H Rosen, Elementary number theory and its applications: Addison-Wesley, 1993. [46]Han Shen, Jian Shen, Muhammad Khurram Khan, and Jong-Hyouk Lee,  Efficient RFID authentication using Elliptic Curve Cryptography for the Internet of Things,  Wireless Personal Communications, pp. 1-14, 2016. DOI=https://doi.org/10.1007/s11277-016-3739-1. [47]Jian Shen, Haowen Tan, Yang Zhang, Xingming Sun, and Yang Xiang,  A new lightweight RFID grouping authentication protocol for multiple tags in mobile environment,  Multimedia Tools and Applications, pp. 1-23, 2017. DOI=https://doi.org/10.1007/s11042-017-4386-6 [48]Boyeon Song and Chris J Mitchell,  RFID authentication protocol for low-cost tags,  in Proceedings of the first ACM conference on Wireless network security, Alexandria, USA, 2008, pp. 140-147. [49]Chunhua Su, Bagus Santoso, Yingjiu Li, Robert Deng, and Xinyi Huang,  Universally composable RFID mutual authentication,  IEEE Transactions on Dependable and Secure Computing, vol. 14, no. 1, pp. 83-94, 2017. [50]Saravanan Sundaresan, Robin Doss, Wanlei Zhou, and Selwyn Piramuthu,  Secure ownership transfer for multi-tag multi-owner passive RFID environment with individual-owner-privacy,  Computer Communications, vol. 55, pp. 112-124, 2015. [51]B Surekha, K Lakshmi Narayana, P Jayaprakash, and Chandra Sekhar Vorugunti,  A realistic lightweight authentication protocol for securing cloud based RFID system,  in Proceedings of 2016 IEEE International Conference on Cloud Computing in Emerging Markets (CCEM), Bangalore, India, 2016, pp. 54-60. [52]Aakanksha Tewari and BB Gupta,  Cryptanalysis of a novel ultra-lightweight mutual authentication protocol for IoT devices using RFID tags,  The Journal of Supercomputing, vol. 73, pp. 1085-1102, 2017. [53]Wei Xie, Lei Xie, Chen Zhang, Quan Zhang, and Chaojing Tang,  Cloud-based RFID authentication,  in Proceedings of 2013 IEEE International Conference on RFID (RFID), Orlando, USA, 2013, pp. 168-175. [54]Ming Hour Yang,  Secure multiple group ownership transfer protocol for mobile RFID,  Electronic Commerce Research and Applications, vol. 11, no. 4, pp. 361-373, 2012. [55]Tzu-Chang Yeh, Chien-Hung Wu, and Yuh-Min Tseng,  Improvement of the RFID authentication scheme based on quadratic residues,  Computer Communications, vol. 34, no. 3, pp. 337-341, 2011. [56]Bianqing Yuan and Jiqiang Liu,  A universally composable secure grouping‐proof protocol for RFID tags,  Concurrency and Computation: Practice and Experience, vol. 28, no. 6, pp. 1872-1883, 2016. [57]Rong Zhang, Liehuang Zhu, Chang Xu, and Yi Yi,  An efficient and secure RFID batch authentication protocol with group tags ownership transfer,  in Proceedings of 2015 IEEE Conference on Collaboration and Internet Computing (CIC), Hangzhou, China, 2015, pp. 168-175. [58]Jingxian Zhou,  A quadratic residue-based lightweight RFID mutual authentication protocol with constant-time identification,  Journal of Communications, vol. 10, no. 2, pp. 117-123, 2015. [59]Wei Zhou, Eun Jung Yoon, and Selwyn Piramuthu,  Simultaneous multi-level RFID tag ownership & transfer in health care environments,  Decision Support Systems, vol. 54, no. 1, pp. 98-108, 2012. [60]Yanjun Zuo,  Changing hands together: a secure group ownership transfer protocol for RFID tags,  in Proceedings of 2010 43rd Hawaii International Conference on System Sciences (HICSS), Honolulu, USA, 2010, pp. 1-10. "
fin_paper_name = open_pkl("./all_pkl/01_paper_name_list.pkl")
fin_keyword = open_pkl("./all_pkl/02_keyword_list.pkl")
fin_abstract = open_pkl("./all_pkl/03_abstract_list.pkl")
fin_content = open_pkl("./all_pkl/04_content_list.pkl")
fin_reference = open_pkl("./all_pkl/05_reference_list.pkl")

def WordSegment_and_write2file(give,rec_dic):
    ws = WS("./data", disable_cuda=False)
    with open('WikiDict_plus_allfieldskeywordsDict.pkl', 'rb') as fp:
        WikiDict_plus_allfieldskeywordsDict = pickle.load(fp)
    fp.close()
    dic1 = construct_dictionary(WikiDict_plus_allfieldskeywordsDict)
    dic2 = {'圖書資訊學系':2, '未出版之碩士論文':2, '未出版之博士論文':2, '服務學習':2, '志願服務':2, '臺灣師範大學':2, '圖書館':2,"國家圖書館":2,"公共圖書館":2}
    rec_dic = rec_dic.split('、')
    for i in rec_dic:
        dic2[i] = 2
    print(dic2)
    force_dic = construct_dictionary(dic2)
    for i in [give]:

            word_sentence_list = ws(
                i, 
                sentence_segmentation = True,
                segment_delimiter_set = {",", "。", ":", "?", "!", ";", "？", "，", "、", " ", "。", "！", "? ", "NULL","\n","\n3000","（","）","=","/"},
                recommend_dictionary = dic1,
                coerce_dictionary = force_dic,
            )
            new_final = []
            for i in word_sentence_list:
                new_i = []
                for j in i:
                    j = remove_punctuation(j)
                    if j != "":
                        if not j.isdigit():
                            new_i.append(j)
                new_final.append(new_i)
            return new_final
conbine_word = [name,keyword,abstract,content,reference]
a = WordSegment_and_write2file(conbine_word,keyword)
print(a)

with open('All_changed.pkl', 'rb') as fp:
    All_list = pickle.load(fp)
fp.close()
with open('All_WS_list.pkl', 'rb') as fp:
    All_list_OLD = pickle.load(fp)
fp.close()

chi_paper_name_WS_list = []
chi_keyword_WS_list= []
abstract_WS_list = []
content_WS_list= []
reference_WS_list= []

for i in range(0,len(All_list)):
    chi_paper_name_WS_list.append(All_list[i][0])
    chi_keyword_WS_list.append(All_list[i][1])
    abstract_WS_list.append(All_list[i][2])
    content_WS_list.append(All_list[i][3])
    reference_WS_list.append(All_list[i][4])

chi_paper_name_WS_list2 = []
chi_keyword_WS_list2= []
abstract_WS_list2 = []
content_WS_list2= []
reference_WS_list2= []

for i in All_list_OLD:
    for j in range(len(All_list_OLD[0])):
        chi_paper_name_WS_list2.append(All_list_OLD[0][j])
        chi_keyword_WS_list2.append(All_list_OLD[1][j])
        abstract_WS_list2.append(All_list_OLD[2][j])
        content_WS_list2.append(All_list_OLD[3][j])
        reference_WS_list2.append(All_list_OLD[4][j])
    break
        
def get_blank_list(list):
    tem_list = []
    for i in list:
        tem = ""
        for j in i :
            tem = tem + j +" "
        tem_list.append(tem)
    # print("確認長度：",len(tem_list))
    return tem_list
def get_tf_idf(corpus,name):
    tem_list = []
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        # print (u"-------這裡輸出第",i,u"類文本的tf-idf權重------")
        tem_tem = []
        for j in range(len(word)):
            if weight[i][j] != 0.0:
                # print(word[j],weight[i][j])
                tem_tem.append([[word[j]],[weight[i][j]]])
        tem_list.append(tem_tem)
    return tem_list
def add (list2,list):
    for i in list2:
        list.append(i)
    print(f"確認長度：{len(list)}")
def sort_tf_idf(name_final):
    sorteded = []
    for i in [name_final]:
        tem_tem = []
        for j in i:
            tem = 0
            for z in j:
                tem += 1
                if tem % 2 == 0:
                    tem_tem.append(z[0])
            #     # print(z)
            
            # print("==========")
        # print(f"trtrtrtr:{tem_tem}")
        # print(sorted(tem_tem, reverse = True))
        sorteded.append(sorted(tem_tem, reverse = True))
    # print(sorteded)
    listlsit = []

    for i in [name_final]:
        dict = {}
        for j in i:
            # print(j[0][0])
            # print(j[1][0])
            dict[j[0][0]] = j[1][0]
        # print("========")
        listlsit.append(dict)
    sortededed = []
    for i in listlsit:
        # print(i)
        print(sorted(i.items(), key=lambda d: d[1], reverse=True))
        sortededed.append(sorted(i.items(), key=lambda d: d[1], reverse=True))
    return sortededed
def get_over_score(list,score):
    list_all_word = []
    stop_word = ["url","html",'http',"www","edu","00年","wwwukoln",'pp',"第一","第二","第三","第四","第五","第六","不詳"]
    for j in list:
        tem = []
        for i in range(0,len(j)):
            if j[i][1] >= score and j[i][0] not in stop_word:
                tem.append(j[i])
        # for i in tem:
        #     print(i)
        for i in tem:
            list_all_word.append(i[0])
        # print("==========")
    print(f"長度是：{len(list_all_word)}")
    return list_all_word

fin_fin = []

add(chi_paper_name_WS_list2,chi_paper_name_WS_list)
chi_paper_name_WS_list.append(a[0])
for i in chi_paper_name_WS_list:
    print(i)
chi_paper_name_WS_BLANK = get_blank_list(chi_paper_name_WS_list)
tem_list = get_tf_idf(chi_paper_name_WS_BLANK,"log_chi_paper_name")
print(f"最終長度：{len(tem_list)}")
print(tem_list[-1])
name_final = tem_list[-1]
name_fin = sort_tf_idf(name_final)
name_fin = get_over_score(name_fin,0.1)
print(name_fin)
fin_fin.append(name_fin)

add(chi_keyword_WS_list2,chi_keyword_WS_list)
chi_keyword_WS_list.append(a[1])
for i in chi_keyword_WS_list:
    print(i)
chi_keyword_WS_BLANK = get_blank_list(chi_keyword_WS_list)
tem_list = get_tf_idf(chi_keyword_WS_BLANK,"log_chi_keyword")
print(f"最終長度：{len(tem_list)}")
print(tem_list[-1])
keyword_final = tem_list[-1]
keyword_fin = sort_tf_idf(keyword_final)
keyword_fin = get_over_score(keyword_fin,0.1)
print(keyword_fin)
fin_fin.append(keyword_fin)

add(abstract_WS_list2,abstract_WS_list)
abstract_WS_list.append(a[2])
for i in abstract_WS_list:
    print(i)
abstract_WS_BLANK = get_blank_list(abstract_WS_list)
tem_list = get_tf_idf(abstract_WS_BLANK,"log_abstract")
print(f"最終長度：{len(tem_list)}")
print(tem_list[-1])
abstract_final = tem_list[-1]
abstract_fin = sort_tf_idf(abstract_final)
abstract_fin = get_over_score(abstract_fin,0.1)
print(abstract_fin)
fin_fin.append(abstract_fin)

add(content_WS_list2,content_WS_list)
content_WS_list.append(a[3])
for i in content_WS_list:
    print(i)
content_WS_BLANK = get_blank_list(content_WS_list)
tem_list = get_tf_idf(content_WS_BLANK,"log_content")
print(f"最終長度：{len(tem_list)}")
print(tem_list[-1])
content_final = tem_list[-1]
content_fin = sort_tf_idf(content_final)
content_fin = get_over_score(content_fin,0.1)
print(content_fin)
fin_fin.append(content_fin)

add(reference_WS_list2,reference_WS_list)
reference_WS_list.append(a[4])
for i in reference_WS_list:
    print(i)
reference_WS_BLANK = get_blank_list(reference_WS_list)
tem_list = get_tf_idf(reference_WS_BLANK,"log_reference")
print(f"最終長度：{len(tem_list)}")
print(tem_list[-1])
reference_final = tem_list[-1]
reference_fin = sort_tf_idf(reference_final)
reference_fin = get_over_score(reference_fin,0.1)
print(reference_fin)
fin_fin.append(reference_fin)

print(fin_fin)

with open("./demo_data/002_demo.pkl", 'wb') as fp:
    pickle.dump(fin_fin, fp)
fp.close()