import json
import re
from itertools import islice

import scrapy


class AcademiaSpider(scrapy.Spider):
    name = "academia"
    url_base = "https://academia.stackexchange.com/"

    # with open(r'C:\Users\annad\Documents\IGA\academia_exchage\Crawler\academia\academia\spiders\most_common_tags.json', 'r') as f:
    #     tags = json.load(f)
    url = []
    tags = ['personal-misconduct', 'research-misconduct', 'sexual-misconduct', 'abuse', 'acknowledgement', 'cheating',
            'discrimination', 'disreputable-publishers','plagiarism', 'self-plagiarism']
    # n_items = list(islice(tags.items(), 30))
    # for k, v in n_items:
    #     #     url.append('https://academia.stackexchange.com/questions/tagged/' + k)
    #     # start_urls = url
    for k in tags:
        url.append('https://academia.stackexchange.com/questions/tagged/' + k)
    start_urls = url

    def parse(self, response):
        yield from self.parseQuestion(response)
        nextPage = response.xpath('//a[@class="s-pagination--item js-pagination-item" and @rel="next"]')
        if nextPage:
            url = nextPage.xpath('.//@href').get()
            # print(url)
            yield scrapy.Request(url=self.url_base + url, callback=self.parse)

    def parseQuestion(self, response):
        # print('parse questions')
        questions = response.xpath('//div[@class="question-summary"]')
        for q in questions:
            questionInfo = self.parseQuestionOverview(q)
            url = q.xpath('.//h3/a/@href').get()
            yield response.follow(url=self.url_base + url, callback=self.parseQuestionDetail,
                                  meta={'questionInfo': questionInfo})

    def parseQuestionOverview(self, question):
        questionInfo = {}
        questionInfo['votes'] = question.xpath('.//span[@class="vote-count-post "]/strong/text()').get()
        questionInfo['answers_count'] = question.xpath('.//div[contains(@class,"status")]/strong/text()').get()
        views = question.xpath('.//div[contains(@class,"views")]/text()').get()
        views = re.sub(' views', '', views)
        views = re.sub('k', '000', views)
        title = question.xpath('.//h3/a/text()').get()
        questionInfo['closed'] = False
        if '[closed]' in title:
            title = re.sub('\[closed\]', '', title)
            questionInfo['closed'] = True
        questionInfo['title'] = title.strip()
        questionInfo['views'] = int(views.strip())
        questionInfo['tags'] = question.xpath('.//div[contains(@class,"tags")]/a/text()').getall()
        asked = question.xpath('.//div[@class="user-action-time"]/span/@title').get()
        questionInfo['asked'] = asked
        return questionInfo

    def parseQuestionDetail(self, response):
        questionInfo = response.meta.get('questionInfo')
        text = response.xpath(
            './/div[@class="question"]/div[@class="post-layout"]/div[@class="postcell post-layout--right"]/div[@class="post-text"]/p')
        text = self.joinParagraph(text)
        questionInfo['post_text'] = text
        comments = response.xpath(
            '//div[@class="question"]/div[@class="post-layout"]/div[@class="post-layout--right"]/div[contains(@class,"comments")]/ul/li/div[contains(@class,"comment-text")]/div[contains(@class,"comment-body")]/span[@class="comment-copy"]')
        if comments:
            questionInfo['comments'] = self.cleanComments(comments)
        else:
            questionInfo['comments'] = None
        questionInfo['answers'] = self.parseAnswers(response)
        yield questionInfo

    def cleanComments(self, comments):
        cleaned = []
        for c in comments:
            cleaned.append(c.xpath('string(.//self::*)').get())
        return cleaned

    def parseAnswers(self, response):
        ans = response.xpath('//div[contains(@class,"answer")]')
        answers = []
        for a in ans:
            answer = {}
            # TODO:hlasy u odpovedi??
            # a.xpath('.//div[@class,"post-layout"]/div[@class,"js-voting-container grid fd-column ai-stretch gs4 fc-black-200"]')
            text = a.xpath(
                './/div[@class="post-layout"]/div[@class="answercell post-layout--right"]/div[@class="post-text"]')
            if text:
                answer['text'] = self.joinParagraph(text)
                answers.append(answer)
            comments = a.xpath(
                './/div[@class="post-layout"]/div[@class="post-layout--right"]/div[contains(@class,"comments")]/ul/li/div[contains(@class,"comment-text")]/div[contains(@class,"comment-body")]/span[@class="comment-copy"]')
            if comments:
                answer['comments'] = self.cleanComments(comments)
            else:
                answer['comments'] = None
        return answers

    def joinParagraph(self, paragraphs):
        text = ''
        for p in paragraphs:
            par = p.xpath('string(.//self::*)').get()
            text += par + '\n'
        return text.strip()
