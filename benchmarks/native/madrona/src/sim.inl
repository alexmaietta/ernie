namespace ernie_bench {

template <typename ArchetypeT> madrona::Entity Engine::makeRenderableEntity() {
    Entity e = makeEntity<ArchetypeT>();
    madrona::render::RenderingSystem::makeEntityRenderable(*this, e);
    return e;
}

}
