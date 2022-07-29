The structure of this framework is based on a survey [1] published by Markus Bayer et al. in which alot of different methods for data augmentation in the NLP subject were compared to eachother. A categorization of this methods was done too and this categorization builds the core structure.
There are four levels of augmentation methods in the data space. Following I will document them.

    Character Level
        Noise -> Introduces errors into data
        Rule based -> Insertion of spelling mistakes, data alterations, entity names and abbreviations
    Word Level
        Noise
            Unigram noising -> Replacing words by different random words
            Blank noising -> Replacing words by "_"
            Syntactic noise -> Shortening, alteration of adjectives
            Semantic noise -> Lexical substitution of synonyms (See next point)
            Random swap (EDA)
            Random deletion (EDA)
            Noise instead of zero-padding
            TF-IDF -> Replace uninformative words by other uninformative
        Synonyms
            Page 9-11 of [1] delivers table with different replacement methods and synonym selections. Choose ones that are delivering positive results and implement 3-4
        Embeddings
            Page 12-14 similar table
            Personally I still have problems totally grasping what this approach is exactly doing, so this will be interesting when I will go into details
        Language Models
            Generate similar words with embeddings, higlhy contextualized
            Page 15 method table
    Phrase Level
        Structure
            POS-Tagging
                Cropping -> shorten sentences by putting focus und object and subject
                Rotation -> Move flexible fragments
            Semantic Text Exchange method
        Interpolation
            Substructure substitution -> Substitute substructures if same tagged label; 4 replacement rules that can be used in any combination
    Document Level
        Translation
            Round-trip translation (RTT) -> Translate word, phrase, sentence or document into one language, then translate back -> Augmented data
        Generative
            Generate new data completely artificial
            Pages 21-22 give some possibilities
            This is the most complicated and new approach, so further information will be written down in an own issues

References

[1]
Markus Bayer, Marc-André Kaufhold, Christian Reuter (2021)
A Survey on Data Augmentation for Text Classification
