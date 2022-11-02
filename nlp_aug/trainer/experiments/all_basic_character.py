from nlp_aug import constants
from nlp_aug.trainer.training_pipelines import basic_character_pipeline, promising_character_techniques_pipeline

basic_character_pipeline(constants.AG_NEWS)
basic_character_pipeline(constants.COLA)
basic_character_pipeline(constants.IMDB)
basic_character_pipeline(constants.ROTTEN)
basic_character_pipeline(constants.SST2)
basic_character_pipeline(constants.SUBJ)
basic_character_pipeline(constants.TREC)

promising_character_techniques_pipeline(constants.AG_NEWS)
promising_character_techniques_pipeline(constants.COLA)
promising_character_techniques_pipeline(constants.IMDB)
promising_character_techniques_pipeline(constants.ROTTEN)
promising_character_techniques_pipeline(constants.SST2)
promising_character_techniques_pipeline(constants.SUBJ)
promising_character_techniques_pipeline(constants.TREC)
