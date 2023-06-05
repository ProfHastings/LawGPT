import spacy

nlp = spacy.load('de_core_news_sm')

text = "Gericht OGH Entscheidungsdatum 25.05.2007 Geschäftszahl 6Ob92/07h Kopf Der Oberste Gerichtshof hat durch den Senatspräsidenten des Obersten Gerichtshofs Dr. Pimmer als Vorsitzenden sowie die Hofrätin des Obersten Gerichtshofs Dr. Schenk und die Hofräte des Obersten Gerichtshofs Dr. Schramm, Dr. Gitschthaler und Univ. Doz. Dr. Kodek in der Firmenbuchsache der im Firmenbuch des Landesgerichts Salzburg zu FN ***** eingetragenen B***** reg.Gen.m.b.H. mit dem Sitz in S***** und der Geschäftsanschrift *****, über den außerordentlichen Revisionsrekurs der Genossenschaft, vertreten durch Dr. Georg Zehetmayer, öffentlicher Notar in Hallein, gegen den Beschluss des Oberlandesgerichts Linz als Rekursgericht vom 21. März 2007, GZ 6 R 31/07g-10, womit der Beschluss des Landesgerichts Salzburg vom 18. Jänner 2007, GZ 24 Fr 6425/06b-4, bestätigt wurde, in nichtöffentlicher Sitzung den Beschluss gefasst: Spruch Dem Revisionsrekurs wird nicht Folge gegeben. Text Begründung:"

doc = nlp(text)

print("Tokens: ", [token.text for token in doc])
