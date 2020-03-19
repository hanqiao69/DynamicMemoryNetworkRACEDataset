import tensorflow.compat.v1 as tf
import json
import tokenization
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

BLANK_TOKEN = '<BLANK>'
MAX_ANS_LEN = 15

def generate_all():
    pass

def pad_collate(batch):
    max_context_sen_len = float('-inf')  # sentence中token数
    max_context_len = float('-inf')  # context中sentence数
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer, options = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer, options)
    return default_collate(batch)

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class InputExample(object):
  """A single training/test example for the RACE dataset."""

  def __init__(self,
               example_id,
               context,
               question,
               answer,
               options):
    self.example_id = example_id
    self.context = context
    self.question = question
    self.answer = answer
    self.options = options

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    l = [
        "id: {}".format(self.example_id),
        "context: {}".format(self.context),
        "question: {}".format(self.question),
        "answer: {}".format(self.answer),
        "options: {}".format(self.options)
    ]
    return ", ".join(l)

class RaceProcessor(object):
  """Processor for the RACE data set."""

  def __init__(self, use_spm, do_lower_case, high_only, middle_only):
    super(RaceProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case
    self.high_only = high_only
    self.middle_only = middle_only

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "train"))

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "dev"))

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "test"))

  def get_labels(self):
    """Gets the list of labels for this data set."""
    return ["A", "B", "C", "D"]

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)

  def read_examples(self, data_dir):
    """Read examples from RACE json files."""
    examples = []
    for level in ["middle", "high"]:
      if level == "middle" and self.high_only: continue
      if level == "high" and self.middle_only: continue
      cur_dir = os.path.join(data_dir, level)

      files = os.listdir(cur_dir)
      for file in files:
        if file != '.DS_Store':

              cur_path = os.path.join(cur_dir, file)
              # print('debug msg: ', cur_path)
              # cur_path = os.path.join(cur_dir, str(file_idx) + ".txt")
              with tf.gfile.Open(cur_path) as f:
                for line in f:
                    cur_data = json.loads(line.strip())

                    answers = cur_data["answers"]
                    options = cur_data["options"]
                    questions = cur_data["questions"]
                    context = self.process_text(cur_data["article"])

                    for i in range(len(answers)):
                        answer_idx = ord(answers[i]) - ord("A")
                        answer = self.process_text(options[i][answer_idx])

                        question = self.process_text(questions[i])

                        option = options[i]

                        if "_" in question:
                            question = question.replace("_", BLANK_TOKEN)

                        examples.append(
                            InputExample(
                                example_id=cur_data["id"],
                                context=context,
                                question=question,
                                answer=answer,
                                options=option
                            )
                        )

    return examples

class RACEDataset(Dataset):
    def __init__(self, mode='train'):
        use_spm = False
        do_lower_case = False
        high_only = False
        middle_only = True
        self.mode = mode
        self.QA = adict()
        self.QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
        self.QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
        self.race_dataset = RaceProcessor(use_spm, do_lower_case, high_only, middle_only)
        self.train_examples = self.word_to_idx_examples(self.race_dataset.get_train_examples('.'))
        self.test_examples = self.word_to_idx_examples(self.race_dataset.get_test_examples('.'))
        self.valid_examples = self.word_to_idx_examples(self.race_dataset.get_dev_examples('.'))

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_examples[0])
        elif self.mode == 'valid':
            return len(self.valid_examples[0])
        elif self.mode == 'test':
            return len(self.test_examples[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers, options = self.train_examples
        elif self.mode == 'valid':
            contexts, questions, answers, options = self.valid_examples
        elif self.mode == 'test':
            contexts, questions, answers, options = self.test_examples
        # print ('index debug: ', index, len(contexts), len(questions), len(answers))
        # return contexts[index], questions[index], answers[index]
        # print('__getitem__ debug: ', contexts[index])
        # print('__getitem__ debug: ', questions[index])
        # print('__getitem__ debug: ', answers[index])
        # print('__getitem__ debug: ', options[index])
        return contexts[index], torch.Tensor(questions[index]), torch.Tensor(answers[index]), torch.Tensor(options[index])
        # return torch.Tensor(contexts[index]), torch.Tensor(questions[index]), torch.Tensor(answers[index])
        # return examples[index].context, examples[index].question, examples[index].answer

    def word_to_idx_examples(self, examples):
        questions = []
        contexts = []
        answers = []
        options = []

        # idx = 0
        for example in examples:
            context_sentences = example.context.lower().split('. ')
            context = [s.split() for s in context_sentences]
            for sentence in context:
                for token in sentence:
                    self.build_vocab(token)
            # context = [self.QA.VOCAB[token] for token in context]
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = example.question.lower().split()
            for token in question:
                self.build_vocab(token)
            question = [self.QA.VOCAB[token] for token in question]

            answer = example.answer.lower().split()
            for token in answer:
                self.build_vocab(token)
            answer = [self.QA.VOCAB[token] for token in answer]
            if len(answer) > MAX_ANS_LEN:
                answer = answer[:MAX_ANS_LEN]
            else:
                answer += [0] * (MAX_ANS_LEN - len(answer))

            option_list = []
            for option in example.options:
                option = option.lower().split()
                for token in option:
                    self.build_vocab(token)
                option = [self.QA.VOCAB[token] for token in option]
                if len(option) > MAX_ANS_LEN:
                    option = option[:MAX_ANS_LEN]
                else:
                    option += [0] * (MAX_ANS_LEN - len(option))
                option_list.append(option)
            # idx += 1
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
            options.append(option_list)

        return (contexts, questions, answers, options)


    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token

# Dataloader Test
if __name__ == '__main__':
    # use_spm = False
    # do_lower_case = False
    # high_only = False
    # middle_only = True
    # race_dataset = RaceProcessor(use_spm, do_lower_case, high_only, middle_only)
    # train_examples = race_dataset.get_train_examples('.')
    # # print(len(train_examples))
    # print(train_examples[1])

    dset_train = RACEDataset(mode='train')
    print('dataset len: ', len(dset_train))
    print('VOCAB size: ', len(dset_train.QA.VOCAB))
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=False, collate_fn=pad_collate)
    idx = 0
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers, options = data
        if idx == 1:
            print('contexts: \n', contexts.shape)
            print('contexts: \n', contexts)
            print('questions: \n', questions.shape)
            print('questions: \n', questions)
            print('answers: \n', answers.shape)
            print('answers: \n', answers)
            print('options: \n', options.shape)
            print('options: \n', options)
            break
        idx += 1

