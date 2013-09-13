#include "Visual.h"

using namespace sge;

static _mouse        *m_mouse;
static _fps          *m_fps;
static _volume2D     *m_volume;
static _viewMatrix   *m_view;
static FreeType      *m_font;
static MainActivity  *m_hAct;



Visual::Visual(GLuint width, GLuint height, MainActivity *hActivity)
{
	m_mouse  = new _mouse;
	m_fps    = new _fps;
	m_volume = new _volume2D;
	m_view   = new _viewMatrix;
	m_font   = new FreeType;
	m_hAct   = hActivity;
};


Visual::~Visual()
{
	SAFE_DELT_PTR(m_mouse);
	SAFE_DELT_PTR(m_fps);
	SAFE_FREE_PTR(m_volume->data);
	SAFE_DELT_PTR(m_volume);
	SAFE_DELT_PTR(m_view);
	m_font->Clean();
	SAFE_DELT_PTR(m_font);
};