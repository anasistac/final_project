
'''This file contains a list of stopwords for use in LDA topic modeling.'''

# Custom movie words
custom_movie_words = {"animation", "scene", "movie", "character", "song", "gonna", "hopps", "lot", "sir", 
                      "mei", "japanse", "japan", "guy", "boy", "girl", "hey", "αrren", "thing", "things"}

# main characters from the disney movies 
disney_main_characters = {"aladdin", "jasmine", "genie", "jafar", "abu", "iago", "sultan",
    "belle", "beast", "gaston", "lumiere", "cogsworth", "mrs_potts", "chip", "lefou",
    "chicken_little", "abby_mallard", "runt", "fish", "buck_cluck",
    "elsa", "anna", "kristoff", "olaf", "sven", "hans", "duke",
    "dumbo", "mrs_jumbo", "timothy", "ringmaster", "crows",
    "hercules", "megara", "phil", "hades", "zeus", "pain", "panic",
    "lilo", "stitch", "nani", "jumba", "pleakley", "cobra_bubbles", "david",
    "moana", "maui", "gramma_tala", "hei_hei", "pua", "te_fiti", "tamatoa",
    "pinocchio", "geppetto", "jiminy_cricket", "blue_fairy", "honest_john", "stromboli",
    "ralph", "vanellope", "felix", "calhoun", "shank", "yesss", "knowsmore", "spamley",
    "pocahontas", "john_smith", "powhatan", "meeko", "flit", "grandmother_willow", "ratcliffe", "nakoma",
    "ariel", "eric", "sebastian", "flounder", "scuttle", "triton", "ursula",
    "simba", "mufasa", "nala", "scar", "timon", "pumbaa", "rafiki", "zazu",
    "woody", "buzz", "jessie", "potato_head", "slinky", "rex", "hamm", "bo_peep", "andy",
    "ralph", "vanellope", "felix", "calhoun", "shank", "yesss", "knowsmore", "spamley",
    "judy", "nick", "bogo", "bellwether", "clawhauser", "flash", "mr_big",
    "rapunzel", "flynn", "gothel", "pascal", "maximus", "toy"
}


# main characters from the ghibli movies
ghibli_main_characters = {
    'arrietty', 'sho',
    'sheeta', 'pazu',
    'umi', 'shun',
    'seita', 'setsuko',
    'kiki', 'jiji', 'tombo',
    'satsuki', 'mei', 'totoro',
    'nausicaa', 'asbel',
    'taeko',
    'ponyo', 'sosuke',
    'porco', 'fio',
    'ashitaka', 'san', 'eboshi',
    'chihiro', 'haku', 'yubaba', 'no_face',
    'arren', 'tenar', 'ged',
    'mahito', 'heron',
    'jiro', 'naoko',
    'anna', 'marnie',
    'shizuku', 'seiji'
}


# Stopwords from standard-mallet-en.txt
stopwords_mallet = {
    "a", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards",
    "again", "against", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although",
    "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone",
    "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are",
    "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but",
    "by", "c", "came", "can", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly",
    "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain",
    "containing", "contains", "corresponding", "could", "course", "currently", "d", "definitely", "described",
    "despite", "did", "different", "do", "does", "doing", "done", "down", "downwards", "during", "e", "each",
    "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even",
    "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except",
    "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly",
    "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "gonna"
    "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "happens", "hardly", "has", "have",
    "having", "he", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i",
    "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
    "inner", "insofar", "instead", "into", "inward", "is", "it", "its", "itself", "j", "just", "k", "keep", "keeps",
    "kept", "know", "knows", "known", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest",
    "let", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
    "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much",
    "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs",
    "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor",
    "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
    "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought",
    "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly",
    "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q",
    "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless",
    "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says",
    "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
    "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "since", "six",
    "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
    "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure",
    "t", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the",
    "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "theres", "thereupon", "these", "they", "think", "third", "this", "thorough", "thoroughly",
    "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took",
    "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under",
    "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful",
    "uses", "using", "usually", "uucp", "v", "value", "various", "very", "via", "viz", "vs", "w", "want",
    "wants", "was", "way", "we", "welcome", "well", "went", "were", "what", "whatever", "when", "whence",
    "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
    "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "willing", "wish", "with", "within", "without", "wonder", "would", "x", "y", "yes", "yet", "you", "your",
    "yours", "yourself", "yourselves", "z", "zero"
}


