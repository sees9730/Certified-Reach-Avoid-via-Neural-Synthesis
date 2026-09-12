import sys

if len(sys.argv) > 1 and sys.argv[1] == 'anim':
    from .animate import main as anim_main
    anim_main(sys.argv[2:])
else:
    from .main import main
    main()
