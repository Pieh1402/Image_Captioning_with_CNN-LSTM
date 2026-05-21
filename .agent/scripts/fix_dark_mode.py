import re

def fix():
    path = 'frontend/templates/index.html'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. fix text-on-surface to have dark:text-white
    content = content.replace('text-on-surface ', 'text-on-surface dark:text-white ')
    content = content.replace('text-on-surface\"', 'text-on-surface dark:text-white\"')
    
    # 2. fix text-on-surface-variant to have dark:text-gray-300
    content = content.replace('text-on-surface-variant ', 'text-on-surface-variant dark:text-gray-300 ')
    content = content.replace('text-on-surface-variant\"', 'text-on-surface-variant dark:text-gray-300\"')
    
    # 3. fix text-secondary to have dark:text-secondary-fixed-dim
    content = content.replace('text-secondary ', 'text-secondary dark:text-secondary-fixed-dim ')
    content = content.replace('text-secondary\"', 'text-secondary dark:text-secondary-fixed-dim\"')
    content = content.replace('dark:text-secondary-fixed-dim dark:text-secondary-fixed-dim', 'dark:text-secondary-fixed-dim')

    # 4. backgrounds
    content = content.replace('bg-surface-variant/30 ', 'bg-surface-variant/30 dark:bg-black/30 ')
    content = content.replace('bg-surface-container-lowest/30 ', 'bg-surface-container-lowest/30 dark:bg-black/40 ')
    
    # 5. gradient title
    content = content.replace('from-primary to-primary-container drop-shadow-sm', 'from-primary to-primary-container dark:from-blue-400 dark:to-indigo-300 drop-shadow-sm')

    # Remove dupes if any
    content = content.replace('dark:text-white dark:text-white', 'dark:text-white')
    content = content.replace('dark:text-gray-300 dark:text-gray-300', 'dark:text-gray-300')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed!")

fix()
