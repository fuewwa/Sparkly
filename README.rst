=======
Sparkly
=======

Sparkly is a very small chat bot for the terminal. Its brain is a neural
network written from scratch in Rust, with no external crates.

There are two modes.

Generative mode (default)
=========================

A character-level language model. It reads the last 40 characters, looks each
one up in an embedding table, passes them through one hidden tanh layer and
predicts the next character with a softmax. A reply is produced one character
at a time, using temperature sampling, until a newline appears.

Training uses plain stochastic gradient descent with a decaying learning rate
and a little input dropout. The training text is ``data/dialogue.txt``:

.. code-block:: text

   you: hello
   sparkly: hello! i am sparkly. what is on your mind?

Everything is lowercase. Characters that never appear in the data are replaced
by spaces in your input.

Expect good answers for phrases close to the training data and noisy,
broken text for anything else. A model this small needs a lot more text to
generalize. Add many more exchanges to ``data/dialogue.txt`` and train again.

Classic mode
============

An intent classifier: a bag-of-words input, one hidden layer and a softmax
over intents. Replies are picked from ready-made lines in ``data/intents.txt``.
It never produces broken text, but it can only say what is written there.

Build
=====

.. code-block:: sh

   cargo build --release

Run
===

.. code-block:: sh

   ./target/release/sparkly                 # generative chat
   ./target/release/sparkly train           # retrain generative model
   ./target/release/sparkly classic         # classic chat
   ./target/release/sparkly classic train   # retrain classic model

On the first start the needed model is trained and saved to ``chat.model`` or
``classic.model`` in the current directory. Later starts load the file.
Generative training takes about a minute. Type ``quit`` to leave.

Data files are read from ``data/`` in the current directory. If a file is
missing, the copy embedded at build time is used.

Classic data format
===================

.. code-block:: text

   [tag]
   a training phrase
   > a reply Sparkly may give

The ``fallback`` intent has only replies. Lines starting with ``#`` are
ignored.

Layout
======

``src/main.rs``
   Argument handling.

``src/chat.rs``
   Generative mode: corpus loading, training, terminal loop.

``src/lm.rs``
   The character-level network: forward pass, training, sampling, saving.

``src/classic.rs``
   Classic mode terminal loop.

``src/model.rs``
   The intent classifier network.

``src/trainer.rs``
   Builds vocabulary and samples for the classifier.

``src/tokenizer.rs``
   Splits text into words for the classifier.

``src/data.rs``
   Parses the intents file.

``src/rng.rs``
   A small xorshift random number generator.
