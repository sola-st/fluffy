from typing import List, Optional, Union
from numpy import ndarray
import torch.nn as nn
import torch
from torchmetrics import Accuracy, F1Score
from torchtext.vocab import Vocab
from transformers import AutoTokenizer

from varclr.data.preprocessor import CodePreprocessor
from varclr.models import urls_pretrained_model
from varclr.models.model import Encoder
import pytorch_lightning as pl

from flows import FlowSpec

class ClassifierNet(pl.LightningModule):
    """
    A simple feedforward network consisting of 2 hidden layers. 
    It predicts the probabilities of each sink type, given the input data (parameter, function names, or other metadata).

    PyTorch Lightning is used to automate most of the training process.
    See https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html.
    """

    def __init__(
            self,
            embedding_dim: int = 768,
            include_fn: bool = False,
            include_doc: bool = False,
            output_label: list[str] = ['CodeInjection','CommandInjection','None','ReflectedXss','TaintedPath'],
            vocab: Optional[Vocab] = None,
            class_weights: Optional[ndarray] = None
    ) -> None:
        super().__init__()

        self.save_hyperparameters()  # save hyperparam passed to `init`

        self.output_label = output_label
        self.vocab = vocab
        
        self.include_fn = include_fn
        if include_fn:
           embedding_dim *= 2
        
        self.include_doc = include_doc
        if include_doc:
            self.lstm_hidden_size = 100
            self.doc_lstm = nn.Sequential(
                nn.Embedding(len(vocab), 100),
                nn.LSTM(input_size=100,
                        hidden_size=self.lstm_hidden_size,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True),
            )
            self.drop = nn.Dropout(p=0.5)
            
            embedding_dim += self.lstm_hidden_size*2 # bi-direction lstm

        # Use VarCLR embedding
        self.embeddings = Encoder.from_pretrained("varclr-codebert")
        self.embeddings.requires_grad_(False)  # freeze this embedding layer, because VarCLR is already pre-trained
        def decor_bert_forward_GPU(model_forward):
            """Monkey patch the original `decor_bert_forward` function in encoder in order to use GPU."""
            processor = CodePreprocessor()
            tokenizer = AutoTokenizer.from_pretrained(
                urls_pretrained_model.PRETRAINED_TOKENIZER
            )

            def tokenize_and_forward(inputs: Union[str, List[str]]) -> torch.Tensor:
                inputs = processor(inputs)
                return_dict = tokenizer(inputs, return_tensors="pt", padding=True)
                return model_forward(
                    # GPU: (nn)
                    return_dict["input_ids"].to(torch.device("cuda")), return_dict["attention_mask"].to(torch.device("cuda"))                    
                )[0].detach()

            return tokenize_and_forward

        self.embeddings.encode = decor_bert_forward_GPU(self.embeddings.forward)

        # Simple feedforward network classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=embedding_dim, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=500, out_features=250),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=250, out_features=len(self.output_label)))
        self.softmax = nn.Softmax()

        # Loss function
        if class_weights is not None:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float))
        else:
            self.loss = nn.CrossEntropyLoss()
        
        # Module to compute accuracy
        self.accuracy = Accuracy(top_k=1)
        self.f1 = F1Score(num_classes=len(self.output_label), average='macro')

    def forward(self, batch) -> torch.Tensor:
        input, sink = batch # batch of ((spec, processed_doc), sink), should be iterable
        emb = self._encode(input)
        pred = self.classifier(emb)
        prob = self.softmax(pred)
        return prob

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        input, sink = batch # batch of ((spec, processed_doc), sink), should be iterable
        emb = self._encode(input)
        pred = self.classifier(emb)        
        loss = self.loss(pred, sink) # (log)softmax layer is already included in the loss nn.CrossEntropyLoss!
        self.log('train_loss', loss)
        self.log("train_loss_epoch", loss, on_step=True, on_epoch=True, prog_bar=True)
        # prob = self.softmax(pred) 
        # f1_score = self.f1(torch.argmax(prob, dim=1), sink)
        # self.log('train_f1_score', f1_score)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1_score = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss, "val_f1_score": f1_score}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc, f1_score = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss, "test_f1_score": f1_score}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        input, sink = batch # batch of ((spec, processed_doc), sink), should be iterable
        emb = self._encode(input)
        pred = self.classifier(emb)
        prob = self.softmax(pred) # (log)softmax layer is already included in the nn.CrossEntropyLoss!
        loss = self.loss(pred, sink)
        # accuracy
        acc = self.accuracy(prob, sink)
        # f1-score
        f1_score = self.f1(torch.argmax(prob, dim=1), sink)
        return loss, acc, f1_score
    
    def _encode(self, input: tuple[list[FlowSpec], torch.Tensor]) -> torch.Tensor:
        """
        Return a Tensor with shape == torch.Size([N, embedding_dim]), where N is the number of spec in this batch.
        """
        if self.include_fn:
            emb = torch.cat((self.embeddings.encode([spec.param.function for spec in input[0]]), self.embeddings.encode(
                [spec.param.parameter for spec in input[0]])), dim=1)
        else:
            emb = self.embeddings.encode(
                [spec.param.parameter for spec in input[0]])
        
        if self.include_doc:
            output, _ = self.doc_lstm(input[1]) # dim: batch_size x text_len x 2*lstm hidden_size 
            # Get the final hidden state of the sequence (from forward LSTM)
            output_forward = output[:, len(output[0])-1, :self.lstm_hidden_size] 
            # Get the beginning hidden state of the sequence (from reverse LSTM)
            output_reverse = output[:, 0, self.lstm_hidden_size:]
            # dim: batch_size x 2*lstm hidden_size 
            output_combined = torch.cat((output_forward, output_reverse), 1)
            text_feature = self.drop(output_combined)
            emb = torch.cat((emb, text_feature), dim=1)
        return emb
