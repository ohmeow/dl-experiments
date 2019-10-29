from fastai.tabular import *
from fastai.text import *
from fastai.text.data import _join_texts


# similar to the "fasta.text.data.pad_collate" except that it is designed to work with MixedTabularLine items,
# where the final thing in an item is the numericalized text ids.
# we need a collate function to ensure a square matrix with the text ids, which will be of variable length.
def mixed_tabular_pad_collate(samples:BatchSamples, 
                              pad_idx:int=1, pad_first:bool=True) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding."

    samples = to_data(samples)
    max_len = max([len(s[0][-1]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
   
    for i,s in enumerate(samples):
        if pad_first: 
            res[i,-len(s[0][-1]):] = LongTensor(s[0][-1])
        else:         
            res[i,:len(s[0][-1]):] = LongTensor(s[0][-1])
            
        # replace the text_ids array (the last thing in the inputs) with the padded tensor matrix
        s[0][-1] = res[i]
              
    # for the inputs, return a list containing 3 elements: a list of cats, a list of conts, and a list of text_ids
    cats = torch.cat([ s[0][0].unsqueeze(0) for s in samples ], 0)
    conts = torch.cat([ s[0][1].unsqueeze(0) for s in samples ], 0)
    texts = torch.cat([ s[0][2].unsqueeze(0) for s in samples ], 0)
    return [cats, conts, texts], tensor([ s[1] for s in samples ])


class MixedTabularLine(TabularLine):
    "Item's that include both tabular data(`conts` and `cats`) and textual data (numericalized `ids`)"
    
    def __init__(self, cats, conts, cat_classes, col_names, txt_ids, txt_cols, txt_string):
        # tabular
        super().__init__(cats, conts, cat_classes, col_names)

        # add the text bits
        self.text_ids = txt_ids
        self.text_cols = txt_cols
        self.text = txt_string
        
        # append numericalted text data to your input (represents your X values that are fed into your model)
        # self.data = [tensor(cats), tensor(conts), tensor(txt_ids)]
        self.data += [ np.array(txt_ids, dtype=np.int64) ]
        self.obj = self.data
        
    def __str__(self):
        res = super().__str__() + f'Text: {self.text}'
        return res


class MixedTabularProcessor(TabularProcessor):
    
    def __init__(self, ds:ItemList=None, procs=None, 
                 tokenizer:Tokenizer=None, chunksize:int=10000,
                 vocab:Vocab=None, max_vocab:int=60000, min_freq:int=2):
        #pdb.set_trace()
        super().__init__(ds, procs)
    
        self.tokenizer, self.chunksize = ifnone(tokenizer, Tokenizer()), chunksize
        
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab, self.max_vocab, self.min_freq = vocab, max_vocab, min_freq
        
    # process a single item in a dataset
    # NOTE: THIS IS METHOD HAS NOT BEEN TESTED AT THIS POINT (WILL COVER IN A FUTURE ARTICLE)
    def process_one(self, item):
        # process tabular data (copied form tabular.data)
        df = pd.DataFrame([item, item])
        for proc in self.procs: proc(df, test=True)
            
        if len(self.cat_names) != 0:
            codes = np.stack([c.cat.codes.values for n,c in df[self.cat_names].items()], 1).astype(np.int64) + 1
        else: 
            codes = [[]]
            
        if len(self.cont_names) != 0:
            conts = np.stack([c.astype('float32').values for n,c in df[self.cont_names].items()], 1)
        else: 
            conts = [[]]
            
        classes = None
        col_names = list(df[self.cat_names].columns.values) + list(df[self.cont_names].columns.values)
        
        # process textual data
        if len(self.text_cols) != 0:
            txt = _join_texts(df[self.text_cols].values, (len(self.text_cols) > 1))
            txt_toks = self.tokenizer._process_all_1(txt)[0]
            text_ids = np.array(self.vocab.numericalize(txt_toks), dtype=np.int64)
        else:
            txt_toks, text_ids = None, [[]]
            
        # return ItemBase
        return MixedTabularLine(codes[0], conts[0], classes, col_names, text_ids, self.txt_cols, txt_toks)
    
    # processes the entire dataset
    def process(self, ds):
        #pdb.set_trace()
        # process tabular data and then set "preprocessed=False" since we still have text data possibly
        super().process(ds)
        ds.preprocessed = False
        
        # process text data from column(s) containing text
        if len(ds.text_cols) != 0:
            texts = _join_texts(ds.inner_df[ds.text_cols].values, (len(ds.text_cols) > 1))

            # tokenize (set = .text)
            tokens = []
            for i in progress_bar(range(0, len(ds), self.chunksize), leave=False):
                tokens += self.tokenizer.process_all(texts[i:i+self.chunksize])
            ds.text = tokens

            # set/build vocab
            if self.vocab is None: self.vocab = Vocab.create(ds.text, self.max_vocab, self.min_freq)
            ds.vocab = self.vocab
            ds.text_ids = [ np.array(self.vocab.numericalize(toks), dtype=np.int64) for toks in ds.text ]
        else:
            ds.text, ds.vocab, ds.text_ids = None, None, []
            
        ds.preprocessed = True


# each "ds" is of type LabelList(Dataset)
class MixedTabularDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs=64, 
               pad_idx=1, pad_first=True, no_check:bool=False, **kwargs) -> DataBunch:
        
        # only thing we're doing here is setting the collate_fn = to our new "pad_collate" method above
        collate_fn = partial(mixed_tabular_pad_collate, pad_idx=pad_idx, pad_first=pad_first)
        
        kwargs['collate_fn'] = collate_fn
        kwargs['num_workers'] = 1
        return super().create(train_ds, valid_ds, test_ds, path=path, bs=bs, no_check=no_check, **kwargs)


class MixedTabularList(TabularList):
    "A custom `ItemList` that merges tabular data along with textual data"
    
    _item_cls = MixedTabularLine
    _processor = MixedTabularProcessor
    _bunch = MixedTabularDataBunch
    
    def __init__(self, items:Iterator, cat_names:OptStrList=None, cont_names:OptStrList=None, 
                 text_cols=None, vocab:Vocab=None, pad_idx:int=1, 
                 procs=None, **kwargs) -> 'MixedTabularList':
        #pdb.set_trace()
        super().__init__(items, cat_names, cont_names, procs, **kwargs)
        
        self.cols = [] if cat_names == None else cat_names.copy()
        if cont_names: self.cols += cont_names.copy()
        if text_cols: self.cols += text_cols.copy()
        
        self.text_cols, self.vocab, self.pad_idx = text_cols, vocab, pad_idx
        
        # add any ItemList state into "copy_new" that needs to be copied each time "new()" is called; 
        # your ItemList acts as a prototype for training, validation, and/or test ItemList instances that
        # are created via ItemList.new()
        self.copy_new += ['text_cols', 'vocab', 'pad_idx']
        
        self.preprocessed = False
        
    # defines how to construct an ItemBase from the data in the ItemList.items array
    def get(self, i):
        if not self.preprocessed: 
            return self.inner_df.iloc[i][self.cols] if hasattr(self, 'inner_df') else self.items[i]
        
        codes = [] if self.codes is None else self.codes[i]
        conts = [] if self.conts is None else self.conts[i]
        text_ids = [] if self.text_ids is None else self.text_ids[i]
        text_string = None if self.text_ids is None else self.vocab.textify(self.text_ids[i])
        
        return self._item_cls(codes, conts, self.classes, self.col_names, text_ids, self.text_cols, text_string)
    
    # this is the method that is called in data.show_batch(), learn.predict() or learn.show_results() 
    # to transform a pytorch tensor back in an ItemBase. 
    # in a way, it does the opposite of calling ItemBase.data. It should take a tensor t and return 
    # the same king of thing as the get method.
    def reconstruct(self, t:Tensor):
        return self._item_cls(t[0], t[1], self.classes, self.col_names, 
                              t[2], self.text_cols, self.vocab.textify(t[2]))
    
    # tells fastai how to display a custom ItemBase when data.show_batch() is called
    def show_xys(self, xs, ys) -> None:
        "Show the `xs` (inputs) and `ys` (targets)."
        from IPython.display import display, HTML
        
        # show tabular
        display(HTML('TABULAR:<br>'))
        super().show_xys(xs, ys)
        
        # show text
        items = [['text_data', 'target']]
        for i, (x,y) in enumerate(zip(xs,ys)):
            res = []
            res += [' '.join([ f'{tok}({self.vocab.stoi[tok]})' 
                              for tok in x.text.split() if (not self.vocab.stoi[tok] == self.pad_idx) ])]
                
            res += [str(y)]
            items.append(res)
            
        col_widths = [90, 1]
        
        display(HTML('TEXT:<br>'))
        display(HTML(text2html_table(items)))
        
    # tells fastai how to display a custom ItemBase when learn.show_results() is called
    def show_xyzs(self, xs, ys, zs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions)."
        from IPython.display import display, HTML
        
        # show tabular
        super().show_xyzs(xs, ys, zs)
        
        # show text
        items = [['text_data','target', 'prediction']]
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            res = []
            res += [' '.join([ f'{tok}({self.vocab.stoi[tok]})'
                              for tok in x.text.split() if (not self.vocab.stoi[tok] == self.pad_idx) ])]
                
            res += [str(y),str(z)]
            items.append(res)
            
        col_widths = [90, 1, 1]
        display(HTML('<br>' + text2html_table(items)))
    
        
    @classmethod
    def from_df(cls, df:DataFrame, cat_names:OptStrList=None, cont_names:OptStrList=None, 
                text_cols=None, vocab=None, procs=None, **kwargs) -> 'ItemList':
        
        return cls(items=range(len(df)), cat_names=cat_names, cont_names=cont_names, 
                   text_cols=text_cols, vocab=vocab, procs=procs, inner_df=df, **kwargs)