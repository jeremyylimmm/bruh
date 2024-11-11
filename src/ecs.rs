use std::usize;
use std::any::*;
use std::collections::HashMap;

pub trait Component {}

#[derive(Copy, Clone)]
pub struct Entity {
  word: u64
}

impl PartialEq<Entity> for Entity {
  fn eq(&self, other: &Entity) -> bool {
    return self.word == other.word;
  }
}

impl Entity {
  fn index(&self) -> usize {
    return (self.word & (u32::MAX as u64)) as usize;
  }

  fn generation(&self) -> u32 {
    return (self.word >> 32) as u32;
  }
}

pub struct World {
  storage: HashMap<TypeId, Box<dyn GenericComponentStorage>>,
  free_list: Vec<u32>,
  generation: Vec<u32>
}

impl World {
  pub fn new() -> Self {
    return Self {
      storage: HashMap::new(),
      free_list: Vec::new(),
      generation: Vec::new()
    }
  }

  pub fn register<T: 'static + Component>(&mut self) {
    let id = TypeId::of::<T>();

    if !self.storage.contains_key(&id) {
      self.storage.insert(id, Box::new(
        ComponentStorage::<T>::new()
      ));
    }
  }

  pub fn create(&mut self) -> Entity {
    if self.free_list.is_empty() {
      self.free_list.push(self.generation.len() as u32);
      self.generation.push(1);
    }

    let idx = self.free_list.pop().unwrap() as u64;
    let gen = self.generation[idx as usize] as u64;

    return Entity {
      word: (gen << 32) | idx
    };
  }

  fn check(&self, e: Entity) -> bool {
    return e.index() < self.generation.len() && e.generation() == self.generation[e.index()];
  }

  pub fn destroy(&mut self, e: Entity) {
    if self.check(e) {
      self.generation[e.index()] += 1;
      self.free_list.push(e.index() as u32);

      for (_, s) in &mut self.storage {
        s.remove(e);
      }
    }
  }

  fn get_storage<T: 'static + Component>(&self) -> &ComponentStorage<T> {
    return self.storage.get(&TypeId::of::<T>()).expect("component not registered").as_any().downcast_ref().unwrap();
  }

  fn get_storage_mut<T: 'static + Component>(&mut self) -> &mut ComponentStorage<T> {
    return self.storage.get_mut(&TypeId::of::<T>()).expect("component not registered").as_any_mut().downcast_mut().unwrap();
  }

  pub fn add<T: 'static + Component>(&mut self, e: Entity, x: T) {
    self.get_storage_mut().add(e, x);
  }

  pub fn remove<T: 'static + Component>(&mut self, e: Entity) {
    self.get_storage_mut::<T>().remove(e);
  }

  pub fn get<T: 'static + Component>(&self, e: Entity) -> Option<&T> {
    return self.get_storage().get(e);
  }

  pub fn get_mut<T: 'static + Component>(&mut self, e: Entity) -> Option<&mut T> {
    return self.get_storage_mut().get_mut(e);
  }
}

trait GenericComponentStorage {
  fn remove(&mut self, e: Entity);
  fn as_any(&self) -> &dyn Any;
  fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct ComponentStorage<T: Component> {
  packed: Vec<(T, Entity)>,
  sparse: Vec<usize>
}

impl<T: Component> ComponentStorage<T> {
  fn new() -> Self {
    return Self {
      sparse: Vec::new(),
      packed: Vec::new()
    }
  }

  fn add(&mut self, e: Entity, x: T) {
    while e.index() >= self.sparse.len() {
      self.sparse.push(usize::MAX);
    }

    if self.sparse[e.index()] == usize::MAX {
      self.sparse[e.index()] = self.packed.len();
      self.packed.push((x, e));
    }
    else {
      self.packed[self.sparse[e.index()]] = (x, e);
    }
  }

  fn get(&self, e: Entity) -> Option<&T> {
    if e.index() < self.sparse.len() {
      let i = self.sparse[e.index()];
      if i != usize::MAX {
        return self.packed.get(i).map(|(x, _)|x);
      }
    }

    return None;
  }

  fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
    if e.index() < self.sparse.len() {
      let i = self.sparse[e.index()];
      if i != usize::MAX {
        return self.packed.get_mut(i).map(|(x, _)|x);
      }
    }

    return None;
  }

  fn has(&self, e: Entity) -> bool {
    return e.index() < self.sparse.len() && self.sparse[e.index()] != usize::MAX;
  }
}

impl<T: 'static + Component> GenericComponentStorage for ComponentStorage<T> {
  fn remove(&mut self, e: Entity) {
    if self.has(e) {
      let last = self.packed.pop().unwrap();

      if last.1 != e {
        let idx = self.sparse[e.index()];
        self.sparse[last.1.index()] = idx;
        self.packed[idx] = last;
      }

      self.sparse[e.index()] = usize::MAX;
    }
  }

  fn as_any(&self) -> &dyn Any {
    return self;
  }

  fn as_any_mut(&mut self) -> &mut dyn Any {
    return self;
  }
}