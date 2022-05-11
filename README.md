# NLPAug
A framework to simplify the usage of common Data Augmentation methods in the NLP context.

# WIP
### Implementation Planning
I will misuse this README at first to document how I want to implement this framework, which dependencies I think I will need and which issues I have to start and complete to progress this project. My general idea is to be kind of agile in the implementation, in a way that I want to work out specific issues first, when I will progress to the domain that should be implemented. But following I will describe which domains I need to implement and I also want to think about the general way in how I want to implement these. 


### Structure
The structure of this framework is based on a survey [[1]](#1) published by Markus Bayer et al. in which alot of different methods for data augmentation in the NLP subject were compared to eachother. A categorization of this methods was done too and this categorization builds the core structure.  
There are four levels of augmentation methods in the data space. Following I will document them.
* Character Level
    * Noise -> Introduces errors into data
    * Rule based -> Insertion of spelling mistakes, data alterations, entity names and abbreviations
* Word Level
    * Noise
        * Unigram noising -> Replacing words by different random words
        * Blank noising -> Replacing words by "_"
        * Syntactic noise -> Shortening, alteration of adjectives
        * Semantic noise -> Lexical substitution of synonyms (See next point)
        * Random swap (EDA) 
        * Random deletion (EDA)
        * Noise instead of zero-padding
        * TF-IDF -> Replace uninformative words by other uninformative
    * Synonyms
    * Embeddings 
    * Language Models
* Phrase Level
    * Structure
    * Interpolation
* Document Level
    * Translation
    * Generative


## References
<a id="1">[1]</a> 
Markus Bayer, Marc-André Kaufhold, Christian Reuter (2021) 
A Survey on Data Augmentation for Text Classification, 6.