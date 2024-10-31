import sys
from fuzzy_search import FuzzySearch


def main(index_dir, stop_words_path):
    search_engine = FuzzySearch(index_dir, stop_words_path)
    _find = search_engine.find_relevant_documents

    while True:
        cmd = input('search query:')
        if (cmd == 'exit()'):
            break

        cmds = cmd.split(':')
        
        if len(cmds[0].split('-')) != 2 :
            print('Input command must be in the format of find-<k>:<query>')
            continue

        k = int(cmds[0].split('-')[1])
        print(_find(cmds[1], k))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    main()