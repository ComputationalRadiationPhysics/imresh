
namespace sdlcommon {


template<class T_ANIMATION>
int SDL_animControl( SDL_Event const & event, T_ANIMATION & anim )
{
    switch ( event.type )
    {
        case SDL_KEYDOWN:
            switch ( event.key.keysym.sym )
            {
                case SDLK_s:
                    anim.step();
                    break;
                case SDLK_PLUS:
                    anim.getRenderFrameDelay() /= 2;
                    break;
                case SDLK_MINUS:
                    anim.getRenderFrameDelay() *= 2;
                    if ( anim.getRenderFrameDelay() <= 0 )
                        anim.getRenderFrameDelay() = 1;
                    break;
                case SDLK_SPACE:
                    anim.togglePause();
                    break;
                default: break;
            }
            break;
        default: break;
    }
    return 0;
}


} // namespace sdlcommon
