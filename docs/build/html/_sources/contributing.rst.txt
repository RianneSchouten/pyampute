Contribution Guidelines
====================================
This document describes the workflow on how to contribute to the pyampute package.


Reporting Bugs
#########################
We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

* Verify that your issue is not being currently addressed by other issues or pull requests.

* Please ensure all code snippets and error messages are formatted in appropriate code blocks. See Creating and highlighting code blocks.

* Please include your operating system type and version number, as well as your Python, pyampute, scikit-learn, numpy versions.

Contributions
#########################
All the contributions to pyampute should linked to issues on Github issue tracker. if you encountered a particular part of the documentation
or code that you want to improve, but there is no related open issue yet, open one first.
This is important since you can first get feedback or pointers from experienced contributors.
To let everyone know you are working on an issue, please leave a comment that states you will work on the issue
(or, if you have the permission, assign yourself to the issue). This avoids double work!

Pull Requests
#########################
All contributions to pyampute should be done via Pull requests. To make a PR to pyampute

* Fork the Pyampute repository.

* create a feature branch to hold your development changes.

* Develop your feature on feature branch and add changes then commit files.

* Follow `these instructions`_ to make a PR to the pyampute repository, this will send email to the authors.
.. _these instructions: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
Pull Requests checklist
#########################
We recommended that your contribution complies with the following rules before you submit a pull request:
* Follow the pep8 style guide. With the following exceptions or additions:

- The max line length is 100 characters instead of 80.

- When creating a multi-line expression with binary operators, break before the operator.

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* An incomplete contribution -- where you expect to do more work before receiving a full review -- should be submitted as a draft. These may be useful to: indicate you are working on something to avoid duplicated work, request broad review of functionality or API, or seek collaborators. Drafts often benefit from the inclusion of a task list in the PR description.

* Add unit tests and examples for any new functionality being introduced.(we use unittest for testing)

* Documentation and high-coverage tests are necessary for enhancements to be accepted. Bug-fixes or new features should be provided with non-regression tests. These tests verify the correct behavior of the fix or feature. In this manner, further modifications on the code base are granted to be consistent with the desired behavior.

(contribution guidelines adapted from openml-python)