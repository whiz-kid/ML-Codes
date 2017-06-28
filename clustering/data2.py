import feedparser
import re

def getWordCounts(url):
	#parse the feed
	d = feedparser.parse(url)
	wc = {}

	#loop over all the entries
	for e in d.entries:
		if 'summary' in e: summary = e.summary
		else: summary = e.description

		#extract all the words
		words = getWords(e.title + " " + summary)
		for word in words:
			wc.setdefault(word,0)
			wc[word] += 1

	return d.feed.title,wc



def getWords(html):
	#remove all the html tags
	text = re.compile(r'<[^>]+>').sub("",html)

	#Split words by all non-alpha characters
	words = re.compile(r'[^A-z^a-z]+').split(text)

	return [word.lower() for word in words if word!=""]



#dictionary for blog title and corresponding word counts
wordCount = {}
apCount = {}
for feedurl in file('feedlist.txt'):
	title,wc = getWordCounts(feedurl)
	wordCount[title] = wc
	for word,count in wc.items():
		apCount.setdefault(word,0)
		if count>1:
			apCount[word]+=1



#removing words whose frequency is <10% and >50%
worldList = []
for w,bc in apCount.items():
	frac = float(bc)/len(fedlist)
	if frac>0.1 and frac<0.5:
		worldList.append(w)



#crteating table for the data
out = file('blogdata.txt','w')
out.write('Blog')
for word in worldList: out.write('\t%s' %word)
out.write('\n')
for blog,wc in wordCount.items():
	out.write(blog)
	for word in worldList:
		if word in wc: out.write('\t%d' %wc[word])
		else: out.write('\t0')
	out.write('\n')


