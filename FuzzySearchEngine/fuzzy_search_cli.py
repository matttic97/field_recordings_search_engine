import argparse
from fuzzy_search import FuzzySearch


def run_search_terminal(index_dir, stop_words_path):
    search_engine = FuzzySearch(index_dir, stop_words_path)
    _find = search_engine.find_relevant_documents

    while True:
        cmd = input('>search query:')
        if (cmd == 'exit()'):
            break

        cmds = cmd.split(':')
        
        if len(cmds[0].split('-')) != 2 :
            print('Input command must be in the format of find-<k>:<query>')
            continue

        k = int(cmds[0].split('-')[1])
        results = _find(cmds[1], k)
        for r in results:
            print(f'Document ID: {r[0]}, Score: {r[1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run fuzzy search engine CLI.")
    parser.add_argument("--index_dir", type=str, required=True, help="Path to the directory with stored indexes.")
    parser.add_argument("--stop_words_path", type=str, default="", help="Path to stop words text file.")

    args = parser.parse_args()

    run_search_terminal(args.index_dir, args.stop_words_path)